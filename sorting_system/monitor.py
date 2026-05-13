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
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor

# === 核心算法依赖 ===
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
# from peft import PeftModel # 移除 LoRA 依赖
from scipy.ndimage import gaussian_filter
import AnomalyCLIP_lib
from prompt_ensemble import AnomalyCLIP_PromptLearner
from AnomalyCLIP_lib.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from utils import normalize

# ================= 配置区域 =================
class Config:
    # 路径配置
    # 修改为您指定的 checkpoint 路径
    CHECKPOINT_PATH = './checkpoints/9_12_4_multiscale/epoch_15.pth'
    # LORA_PATH = './checkpoint_lora2/lora_epoch_1' # 移除 LoRA 路径
    BASE_MODEL = "ViT-L/14@336px"
    
    # 参数配置
    INPUT_SIZE = 518
    MODEL_PARAMS = {
        "Prompt_length": 12, 
        "learnabel_text_embedding_depth": 9, 
        "learnabel_text_embedding_length": 4
    }
    SIGMA = 4
    
    # 默认硬件
    CAMERA_INDEX = 0 # 默认尝试1，失败会尝试0

# ================= 核心检测类 (逻辑层) =================
class AnomalyDetector:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.preprocess = None
        self.text_features = None
        self.prompt_learner = None

    def load_model(self):
        """加载模型，耗时操作"""
        print(f"[Init] Loading Base Model: {Config.BASE_MODEL}...")
        self.model, _ = AnomalyCLIP_lib.load(Config.BASE_MODEL, device=self.device, design_details=Config.MODEL_PARAMS)
        self.model.visual.DAPM_replace(DPAM_layer=20)
        
        # === 移除 LoRA 加载逻辑 ===
        # print(f"[Init] Loading LoRA: {Config.LORA_PATH}...")
        # self.model.visual = PeftModel.from_pretrained(self.model.visual, Config.LORA_PATH)
        
        self.model.eval()
        self.model.to(self.device)
        
        print(f"[Init] Loading Prompt Learner: {Config.CHECKPOINT_PATH}...")
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
            CenterCrop(Config.INPUT_SIZE),
            ToTensor(),
            Normalize(OPENAI_DATASET_MEAN, OPENAI_DATASET_STD)
        ])
        print("[Init] Ready.")

    def predict(self, cv2_img):
        img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
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
                # get_similarity_map 返回 [B, H, W, 2]
                similarity_map = AnomalyCLIP_lib.get_similarity_map(similarity[:, 1:, :], Config.INPUT_SIZE)
                
                # === 按照 test.py 标准修改热力图计算逻辑 ===
                # 公式: (异常图 + (1 - 正常图)) / 2.0
                anomaly_map = (similarity_map[..., 1] + 1 - similarity_map[..., 0]) / 2.0
                similarity_map_list.append(anomaly_map)

            # === 按照 test.py 标准使用 sum (求和) 而非 mean ===
            # stack后 shape: [Layers, B, H, W] -> sum(dim=0) -> [B, H, W]
            similarity_map = torch.stack(similarity_map_list, dim=0).sum(dim=0)
            
            # 取出第一个样本
            similarity_map = similarity_map[0].cpu().numpy()
            anomaly_map = gaussian_filter(similarity_map, sigma=Config.SIGMA)
            
        return anomaly_score, anomaly_map

# ================= 工作线程 (处理摄像头和推理) =================
class DetectorThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray) # 发送图像给UI
    result_signal = pyqtSignal(str, float, np.ndarray) # 发送结果 (Status, Score, Heatmap)
    status_signal = pyqtSignal(str) # 加载状态

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.detector = None
        self.interval = 10
        self.threshold = 0.65
        self.camera_index = Config.CAMERA_INDEX
        self.is_detecting = False # 是否开启检测循环

    def run(self):
        # 1. 初始化模型
        self.status_signal.emit("正在加载模型... (约需10秒)")
        try:
            self.detector = AnomalyDetector()
            self.detector.load_model()
            self.status_signal.emit("模型加载完成")
        except Exception as e:
            self.status_signal.emit(f"模型加载失败: {str(e)}")
            return

        # 2. 打开摄像头
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0) # 尝试备用索引
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        last_check_time = time.time()
        last_heatmap_vis = None # 缓存上一次的热力图
        show_heatmap_until = 0

        while self._run_flag:
            ret, frame = cap.read()
            if not ret:
                self.status_signal.emit("无法读取摄像头")
                time.sleep(1)
                continue
            
            # 取消镜像翻转，保持实际第一视角
            # frame = cv2.flip(frame, 1) 
            display_frame = frame.copy()
            current_time = time.time()

            # === 检测逻辑 ===
            if self.is_detecting and (current_time - last_check_time >= self.interval):
                self.status_signal.emit("正在检测...")
                
                # 裁剪中心 ROI (使用 Config.INPUT_SIZE = 518)
                h, w = frame.shape[:2]
                # 修改：使用更小的框 (518x518)，而不是 min(h, w) (720x720)
                crop_size = Config.INPUT_SIZE
                if crop_size > min(h, w): crop_size = min(h, w)
                
                sx = w//2 - crop_size//2
                sy = h//2 - crop_size//2
                roi = frame[sy:sy+crop_size, sx:sx+crop_size]

                # 推理
                try:
                    score, raw_heatmap = self.detector.predict(roi)
                    status = "NG" if score > self.threshold else "OK"
                    
                    # 处理热力图可视化
                    # roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    # 调整热力图大小到 ROI 大小
                    raw_heatmap_resized = cv2.resize(raw_heatmap, (roi.shape[1], roi.shape[0]))
                    heatmap_norm = normalize(raw_heatmap_resized)
                    
                    # 生成彩色热力图
                    heatmap_vis = (heatmap_norm * 255).astype(np.uint8)
                    heatmap_vis = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_JET)
                    # 融合 (50% 原图 + 50% 热力图)
                    overlay = cv2.addWeighted(roi, 0.6, heatmap_vis, 0.4, 0)
                    
                    last_heatmap_vis = overlay
                    show_heatmap_until = current_time + 3.0 # 显示结果3秒
                    last_check_time = current_time
                    
                    # 发送结果信号
                    self.result_signal.emit(status, score, overlay)
                    self.status_signal.emit(f"检测完成: {status}")
                    
                except Exception as e:
                    print(f"Inference Error: {e}")

            # === 绘制逻辑 ===
            # 绘制中心框
            h, w = frame.shape[:2]
            # 同样使用更小的框显示
            crop_size = Config.INPUT_SIZE
            if crop_size > min(h, w): crop_size = min(h, w)
            
            sx = w//2 - crop_size//2
            sy = h//2 - crop_size//2
            
            # 如果在展示时间内，显示热力图在中心
            if current_time < show_heatmap_until and last_heatmap_vis is not None:
                display_frame[sy:sy+crop_size, sx:sx+crop_size] = last_heatmap_vis
                cv2.rectangle(display_frame, (sx, sy), (sx+crop_size, sy+crop_size), (0, 255, 255), 3)
            else:
                # 正常显示框
                color = (0, 255, 0) if self.is_detecting else (100, 100, 100)
                cv2.rectangle(display_frame, (sx, sy), (sx+crop_size, sy+crop_size), color, 2)
            
            self.change_pixmap_signal.emit(display_frame)
            time.sleep(0.03) # 限制在 ~30FPS

        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

    def update_params(self, threshold, interval):
        self.threshold = threshold
        self.interval = interval

    def toggle_detection(self, active):
        self.is_detecting = active

# ================= 主界面 UI =================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AnomalyCLIP 智能工业质检系统")
        self.resize(1400, 850)
        
        # 样式表
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; color: #ffffff; }
            QLabel { color: #e0e0e0; font-family: 'Microsoft YaHei'; }
            QGroupBox { border: 1px solid #3e3e3e; border-radius: 5px; margin-top: 10px; font-weight: bold; color: #aaaaaa; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }
            QPushButton { background-color: #0078d7; color: white; border: none; border-radius: 5px; padding: 10px; font-weight: bold; font-size: 14px; }
            QPushButton:hover { background-color: #1084e3; }
            QPushButton:disabled { background-color: #333333; color: #777777; }
            QListWidget { background-color: #252526; border: 1px solid #3e3e3e; border-radius: 5px; color: #dddddd; }
            QProgressBar { border: 1px solid #3e3e3e; border-radius: 5px; text-align: center; }
            QProgressBar::chunk { background-color: #0078d7; }
        """)

        self.init_ui()
        self.start_thread()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)

        # === 左侧：视频显示区 ===
        video_group = QGroupBox("实时监控 / 缺陷热力图")
        video_layout = QVBoxLayout()
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setText("等待摄像头启动...")
        self.image_label.setStyleSheet("background-color: #000; border-radius: 5px;")
        self.image_label.setMinimumSize(960, 540)
        video_layout.addWidget(self.image_label)
        video_group.setLayout(video_layout)
        
        # === 右侧：控制与信息区 ===
        control_layout = QVBoxLayout()
        control_layout.setSpacing(20)

        # 1. 状态显示
        status_group = QGroupBox("当前状态")
        status_layout = QVBoxLayout()
        
        self.lbl_result = QLabel("等待检测")
        self.lbl_result.setAlignment(Qt.AlignCenter)
        self.lbl_result.setStyleSheet("font-size: 36px; font-weight: bold; color: #888888; background-color: #2d2d2d; border-radius: 8px; padding: 15px;")
        
        self.lbl_score = QLabel("异常分数: 0.0000")
        self.lbl_score.setStyleSheet("font-size: 16px; color: #cccccc;")
        
        self.score_bar = QProgressBar()
        self.score_bar.setRange(0, 100)
        self.score_bar.setValue(0)
        self.score_bar.setFixedHeight(10)
        
        status_layout.addWidget(self.lbl_result)
        status_layout.addWidget(self.lbl_score)
        status_layout.addWidget(self.score_bar)
        status_group.setLayout(status_layout)

        # 2. 参数设置
        settings_group = QGroupBox("参数设置")
        settings_layout = QVBoxLayout()
        
        # 阈值
        self.lbl_thresh = QLabel("判定阈值: 0.65")
        self.slider_thresh = QSlider(Qt.Horizontal)
        self.slider_thresh.setRange(0, 100)
        self.slider_thresh.setValue(65)
        self.slider_thresh.valueChanged.connect(self.on_param_change)
        
        # 间隔
        self.lbl_interval = QLabel("检测间隔: 10秒")
        self.slider_interval = QSlider(Qt.Horizontal)
        self.slider_interval.setRange(1, 30)
        self.slider_interval.setValue(10)
        self.slider_interval.valueChanged.connect(self.on_param_change)
        
        settings_layout.addWidget(self.lbl_thresh)
        settings_layout.addWidget(self.slider_thresh)
        settings_layout.addWidget(self.lbl_interval)
        settings_layout.addWidget(self.slider_interval)
        settings_group.setLayout(settings_layout)

        # 3. 控制按钮
        self.btn_start = QPushButton("开始自动检测")
        self.btn_start.setCheckable(True)
        self.btn_start.clicked.connect(self.toggle_detection)
        self.btn_start.setEnabled(False) # 模型加载完才启用

        # 4. 历史记录
        log_group = QGroupBox("检测记录")
        log_layout = QVBoxLayout()
        self.list_log = QListWidget()
        log_layout.addWidget(self.list_log)
        log_group.setLayout(log_layout)
        
        # 底部状态栏
        self.lbl_status = QLabel("系统初始化中...")
        self.lbl_status.setStyleSheet("color: #666666; font-size: 12px;")

        # 添加到右侧布局
        control_layout.addWidget(status_group)
        control_layout.addWidget(settings_group)
        control_layout.addWidget(self.btn_start)
        control_layout.addWidget(log_group)
        control_layout.addWidget(self.lbl_status)
        control_layout.addStretch()

        # 组合主布局
        main_layout.addWidget(video_group, stretch=3)
        main_layout.addLayout(control_layout, stretch=1)

    def start_thread(self):
        self.thread = DetectorThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.result_signal.connect(self.update_result)
        self.thread.status_signal.connect(self.update_status)
        self.thread.start()

    def update_image(self, cv_img):
        """将OpenCV图像转换为QPixmap显示"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio)
        self.image_label.setPixmap(QPixmap.fromImage(p))

    def update_result(self, status, score, heatmap):
        # 更新大字状态
        if status == "NG":
            self.lbl_result.setText("NG - 异常")
            self.lbl_result.setStyleSheet("font-size: 36px; font-weight: bold; color: #ffffff; background-color: #d32f2f; border-radius: 8px; padding: 15px;")
        else:
            self.lbl_result.setText("OK - 正常")
            self.lbl_result.setStyleSheet("font-size: 36px; font-weight: bold; color: #ffffff; background-color: #388e3c; border-radius: 8px; padding: 15px;")
        
        # 更新分数条
        self.lbl_score.setText(f"异常分数: {score:.4f}")
        self.score_bar.setValue(int(score * 100))
        
        # 添加记录到列表
        timestamp = datetime.now().strftime("%H:%M:%S")
        item_text = f"[{timestamp}] {status} | Score: {score:.3f}"
        item = QListWidgetItem(item_text)
        if status == "NG":
            item.setForeground(QColor("#ff5555"))
        else:
            item.setForeground(QColor("#55ff55"))
        self.list_log.insertItem(0, item) # 插入到最上面

    def update_status(self, text):
        self.lbl_status.setText(text)
        if text == "模型加载完成":
            self.btn_start.setEnabled(True)
            self.btn_start.setText("开始自动检测")

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
            # 重置状态显示
            self.lbl_result.setText("等待检测")
            self.lbl_result.setStyleSheet("font-size: 36px; font-weight: bold; color: #888888; background-color: #2d2d2d; border-radius: 8px; padding: 15px;")

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())