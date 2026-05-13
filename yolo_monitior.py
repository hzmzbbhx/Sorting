import sys
import cv2
import time
import os
import torch
import numpy as np
from datetime import datetime
from PIL import Image

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, 
                             QVBoxLayout, QHBoxLayout, QPushButton, QSlider, 
                             QGroupBox, QListWidget, QListWidgetItem, QProgressBar, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QColor

# === 核心算法依赖 ===
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
from peft import PeftModel
from scipy.ndimage import gaussian_filter
import AnomalyCLIP_lib
from prompt_ensemble import AnomalyCLIP_PromptLearner
from AnomalyCLIP_lib.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from utils import normalize

# === YOLO 依赖 ===
try:
    from ultralytics import YOLO
except ImportError:
    print("请安装 ultralytics: pip install ultralytics")
    sys.exit(1)

# ================= 配置区域 =================
class Config:
    # 1. AnomalyCLIP 路径
    CHECKPOINT_PATH = './checkpoint_lora_pokerback/best_model.pth'
    LORA_PATH = './checkpoint_lora_pokerback/best_lora'
    BASE_MODEL = "ViT-L/14@336px"
    
    # 2. YOLO 配置
    YOLO_MODEL_PATH = 'D:/test/AnomalyCLIP/runs/detect/train3/weights/best.pt' 
    YOLO_CONF_THRESH = 0.5  # YOLO 认为它是"牌"的置信度
    
    # 3. 参数配置
    INPUT_SIZE = 518
    MODEL_PARAMS = {
        "Prompt_length": 12, 
        "learnabel_text_embedding_depth": 9, 
        "learnabel_text_embedding_length": 4
    }
    SIGMA = 4
    
    # 4. 硬件
    CAMERA_INDEX = 0

# ================= 核心检测类 (逻辑层) =================
class AnomalyDetector:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.preprocess = None
        self.text_features = None
        self.prompt_learner = None
        self.yolo_model = None

    def load_model(self):
        # 1. Load YOLO
        print(f"[Init] Loading YOLO: {Config.YOLO_MODEL_PATH}...")
        self.yolo_model = YOLO(Config.YOLO_MODEL_PATH)
        
        # 2. Load AnomalyCLIP
        print(f"[Init] Loading Base Model: {Config.BASE_MODEL}...")
        self.model, _ = AnomalyCLIP_lib.load(Config.BASE_MODEL, device=self.device, design_details=Config.MODEL_PARAMS)
        self.model.visual.DAPM_replace(DPAM_layer=20)
        
        print(f"[Init] Loading LoRA: {Config.LORA_PATH}...")
        self.model.visual = PeftModel.from_pretrained(self.model.visual, Config.LORA_PATH)
        self.model.eval()
        self.model.to(self.device)
        
        print(f"[Init] Loading Prompt Learner...")
        self.prompt_learner = AnomalyCLIP_PromptLearner(self.model, Config.MODEL_PARAMS)
        checkpoint = torch.load(Config.CHECKPOINT_PATH, map_location='cpu')
        state_dict = checkpoint["prompt_learner"] if "prompt_learner" in checkpoint else checkpoint
        self.prompt_learner.load_state_dict(state_dict)
        self.prompt_learner.to(self.device)
        self.prompt_learner.eval()
        
        print("[Init] Pre-computing features...")
        with torch.no_grad():
            prompts, tokenized_prompts, compound_prompts_text = self.prompt_learner(cls_id=None)
            text_features = self.model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
            text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)
            self.text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
        self.preprocess = Compose([
            Resize((Config.INPUT_SIZE, Config.INPUT_SIZE), interpolation=InterpolationMode.BICUBIC),
            ToTensor(),
            Normalize(OPENAI_DATASET_MEAN, OPENAI_DATASET_STD)
        ])
        print("[Init] Ready.")

    def detect_objects_yolo(self, cv2_img):
        """返回所有检测到的物体框"""
        results = self.yolo_model(cv2_img, conf=Config.YOLO_CONF_THRESH, verbose=False)
        detected_boxes = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # x1, y1, x2, y2
                coords = box.xyxy[0].cpu().numpy().astype(int)
                detected_boxes.append(coords)
        return detected_boxes

    def predict_anomaly(self, crop_img):
        img_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        img_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features, patch_features = self.model.encode_image(
                img_tensor, feature_list=[6, 12, 18, 24], DPAM_layer=20
            )
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            text_probs = image_features.unsqueeze(1) @ self.text_features.permute(0, 2, 1)
            text_probs = (text_probs / 0.07).softmax(-1)
            anomaly_score = text_probs[:, 0, 1].item()

            similarity_map_list = []
            for patch_feature in patch_features:
                patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)
                similarity, _ = AnomalyCLIP_lib.compute_similarity(patch_feature, self.text_features[0])
                similarity_map = AnomalyCLIP_lib.get_similarity_map(similarity[:, 1:, :], Config.INPUT_SIZE)
                similarity_map_list.append(similarity_map)

            similarity_map = torch.stack(similarity_map_list, dim=0).mean(dim=0)
            similarity_map = similarity_map[0, :, :, 1].cpu().numpy()
            anomaly_map = gaussian_filter(similarity_map, sigma=Config.SIGMA)
            
        return anomaly_score, anomaly_map

# ================= 工作线程 =================
class DetectorThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray) 
    result_signal = pyqtSignal(list) # 修改：发送列表，包含所有物体的结果
    status_signal = pyqtSignal(str) 

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.detector = None
        self.interval = 5 
        # 这里改回 0.65，或者你可以根据实际效果在界面上拉动滑块调整
        self.threshold = 0.65 
        self.camera_index = Config.CAMERA_INDEX
        self.is_detecting = False 

    def run(self):
        self.status_signal.emit("正在加载模型... (YOLO + CLIP)")
        try:
            self.detector = AnomalyDetector()
            self.detector.load_model()
            self.status_signal.emit("模型加载完成")
        except Exception as e:
            self.status_signal.emit(f"模型加载失败: {str(e)}")
            return

        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened(): cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        last_check_time = time.time()
        
        # 缓存上一帧的绘制结果（用于在非检测帧保持显示）
        cached_results = [] # 格式: [{'box':(x1,y1,x2,y2), 'status':'OK', 'score':0.1, 'overlay':img}, ...]
        show_until = 0

        while self._run_flag:
            ret, frame = cap.read()
            if not ret:
                self.status_signal.emit("无法读取摄像头")
                time.sleep(1)
                continue
            
            display_frame = frame.copy()
            current_time = time.time()

            # === 检测逻辑 ===
            if self.is_detecting and (current_time - last_check_time >= self.interval):
                self.status_signal.emit("正在多目标检测...")
                
                # 1. YOLO 找所有框
                boxes = self.detector.detect_objects_yolo(frame)
                
                if len(boxes) > 0:
                    new_results = []
                    ng_count = 0
                    
                    for box in boxes:
                        x1, y1, x2, y2 = box
                        # 边缘保护
                        h, w = frame.shape[:2]
                        pad = 10
                        x1 = max(0, x1-pad); y1 = max(0, y1-pad)
                        x2 = min(w, x2+pad); y2 = min(h, y2+pad)
                        
                        roi = frame[y1:y2, x1:x2]
                        if roi.size == 0: continue
                        
                        # 2. 对每个框跑 CLIP
                        try:
                            score, raw_map = self.detector.predict_anomaly(roi)
                            status = "NG" if score > self.threshold else "OK"
                            if status == "NG": ng_count += 1
                            
                            # 3. 生成局部热力图
                            raw_map_res = cv2.resize(raw_map, (roi.shape[1], roi.shape[0]))
                            h_norm = normalize(raw_map_res)
                            h_vis = (h_norm * 255).astype(np.uint8)
                            h_vis = cv2.applyColorMap(h_vis, cv2.COLORMAP_JET)
                            roi_overlay = cv2.addWeighted(roi, 0.6, h_vis, 0.4, 0)
                            
                            new_results.append({
                                'box': (x1, y1, x2, y2),
                                'status': status,
                                'score': score,
                                'overlay': roi_overlay
                            })
                            
                        except Exception as e:
                            print(e)
                    
                    cached_results = new_results
                    show_until = current_time + 3.0
                    last_check_time = current_time
                    
                    # 发送总结果给 UI 列表
                    summary_text = f"检测到 {len(boxes)} 个目标, {ng_count} NG"
                    self.status_signal.emit(summary_text)
                    self.result_signal.emit(new_results) # 发送列表数据用于记录
                    
                else:
                    self.status_signal.emit("视野内无目标")
                    cached_results = []

            # === 绘制逻辑 (支持多目标) ===
            if current_time < show_until and len(cached_results) > 0:
                for res in cached_results:
                    x1, y1, x2, y2 = res['box']
                    status = res['status']
                    score = res['score']
                    overlay = res['overlay']
                    
                    # 1. 贴回热力图
                    # 检查尺寸防止 crash
                    th, tw = overlay.shape[:2]
                    if (y2-y1) == th and (x2-x1) == tw:
                        display_frame[y1:y2, x1:x2] = overlay
                    
                    # 2. 画框和文字
                    color = (0, 0, 255) if status == "NG" else (0, 255, 0)
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    
                    label = f"{status} {score:.2f}"
                    # 文字背景条，防止看不清
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(display_frame, (x1, y1 - 20), (x1 + tw, y1), color, -1)
                    cv2.putText(display_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            self.change_pixmap_signal.emit(display_frame)
            time.sleep(0.03)

        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

    def update_params(self, threshold, interval):
        self.threshold = threshold
        self.interval = interval

    def toggle_detection(self, active):
        self.is_detecting = active

# ================= UI 部分 =================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AnomalyCLIP + YOLO 多目标质检系统")
        self.resize(1400, 850)
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; color: #ffffff; }
            QLabel { color: #e0e0e0; font-family: 'Microsoft YaHei'; }
            QGroupBox { border: 1px solid #3e3e3e; border-radius: 5px; margin-top: 10px; font-weight: bold; color: #aaaaaa; }
            QPushButton { background-color: #0078d7; color: white; border: none; border-radius: 5px; padding: 10px; font-weight: bold; }
            QListWidget { background-color: #252526; border: 1px solid #3e3e3e; border-radius: 5px; color: #dddddd; }
        """)
        self.init_ui()
        self.start_thread()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)

        video_group = QGroupBox("实时监控")
        video_layout = QVBoxLayout()
        self.image_label = QLabel("加载中...")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #000;")
        self.image_label.setMinimumSize(960, 540)
        video_layout.addWidget(self.image_label)
        video_group.setLayout(video_layout)
        
        control_layout = QVBoxLayout()
        
        # 状态
        self.lbl_result = QLabel("等待检测")
        self.lbl_result.setStyleSheet("font-size: 24px; font-weight: bold; color: #888; background-color: #2d2d2d; padding: 10px; border-radius: 5px;")
        self.lbl_result.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(self.lbl_result)

        # 设置
        settings_group = QGroupBox("参数设置")
        s_layout = QVBoxLayout()
        
        self.lbl_thresh = QLabel("判定阈值: 0.65")
        self.slider_thresh = QSlider(Qt.Horizontal)
        self.slider_thresh.setRange(0, 100)
        self.slider_thresh.setValue(65)
        self.slider_thresh.valueChanged.connect(self.on_param_change)
        
        self.lbl_interval = QLabel("检测间隔: 5秒")
        self.slider_interval = QSlider(Qt.Horizontal)
        self.slider_interval.setRange(1, 30)
        self.slider_interval.setValue(5)
        self.slider_interval.valueChanged.connect(self.on_param_change)
        
        s_layout.addWidget(self.lbl_thresh)
        s_layout.addWidget(self.slider_thresh)
        s_layout.addWidget(self.lbl_interval)
        s_layout.addWidget(self.slider_interval)
        settings_group.setLayout(s_layout)
        control_layout.addWidget(settings_group)

        self.btn_start = QPushButton("开始自动检测")
        self.btn_start.setCheckable(True)
        self.btn_start.clicked.connect(self.toggle_detection)
        self.btn_start.setEnabled(False)
        control_layout.addWidget(self.btn_start)

        log_group = QGroupBox("检测日志")
        l_layout = QVBoxLayout()
        self.list_log = QListWidget()
        l_layout.addWidget(self.list_log)
        log_group.setLayout(l_layout)
        control_layout.addWidget(log_group)
        
        self.lbl_status = QLabel("初始化中...")
        control_layout.addWidget(self.lbl_status)

        main_layout.addWidget(video_group, stretch=3)
        main_layout.addLayout(control_layout, stretch=1)

    def start_thread(self):
        self.thread = DetectorThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.result_signal.connect(self.update_log)
        self.thread.status_signal.connect(self.update_status)
        self.thread.start()

    def update_image(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        convert_to_Qt_format = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio)
        self.image_label.setPixmap(QPixmap.fromImage(p))

    def update_log(self, results_list):
        # 处理多目标结果的日志显示
        timestamp = datetime.now().strftime("%H:%M:%S")
        ng_items = [r for r in results_list if r['status'] == 'NG']
        
        if len(ng_items) > 0:
            # 如果有 NG，重点显示 NG
            self.lbl_result.setText(f"发现 {len(ng_items)} 个 NG")
            self.lbl_result.setStyleSheet("font-size: 24px; background-color: #d32f2f; color: white;")
            
            for item in ng_items:
                log_txt = f"[{timestamp}] NG | Score: {item['score']:.3f}"
                list_item = QListWidgetItem(log_txt)
                list_item.setForeground(QColor("#ff5555"))
                self.list_log.insertItem(0, list_item)
        else:
            # 全是 OK
            self.lbl_result.setText(f"全部 OK ({len(results_list)}个)")
            self.lbl_result.setStyleSheet("font-size: 24px; background-color: #388e3c; color: white;")
            
            log_txt = f"[{timestamp}] All {len(results_list)} OK"
            list_item = QListWidgetItem(log_txt)
            list_item.setForeground(QColor("#55ff55"))
            self.list_log.insertItem(0, list_item)

    def update_status(self, text):
        self.lbl_status.setText(text)
        if text == "模型加载完成":
            self.btn_start.setEnabled(True)

    def on_param_change(self):
        thresh = self.slider_thresh.value() / 100.0
        interval = self.slider_interval.value()
        self.lbl_thresh.setText(f"判定阈值: {thresh:.2f}")
        self.lbl_interval.setText(f"检测间隔: {interval}秒")
        if self.thread:
            self.thread.update_params(thresh, interval)

    def toggle_detection(self):
        if self.btn_start.isChecked():
            self.btn_start.setText("停止检测")
            self.btn_start.setStyleSheet("background-color: #d32f2f;")
            self.thread.toggle_detection(True)
        else:
            self.btn_start.setText("开始自动检测")
            self.btn_start.setStyleSheet("background-color: #0078d7;")
            self.thread.toggle_detection(False)
            self.lbl_result.setText("等待检测")
            self.lbl_result.setStyleSheet("background-color: #2d2d2d; color: #888;")

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())