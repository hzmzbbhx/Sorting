import sys
import cv2
import time
import socket
import threading
import torch
import numpy as np
from datetime import datetime
from PIL import Image

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, 
                             QVBoxLayout, QHBoxLayout, QPushButton, QSlider, 
                             QGroupBox, QListWidget, QListWidgetItem, QLineEdit, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QColor

# === 核心算法依赖 ===
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode
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

# ================= 配置区域 (请根据实际情况修改) =================
class Config:
    # 1. 模型路径
    CHECKPOINT_PATH = './checkpoint_lora_pokerback/best_model.pth'
    LORA_PATH = './checkpoint_lora_pokerback/best_lora'
    BASE_MODEL = "ViT-L/14@336px"
    YOLO_MODEL_PATH = 'D:/test/AnomalyCLIP/runs/detect/train3/weights/best.pt' 
    YOLO_CONF_THRESH = 0.5
    
    # 2. 模型参数
    INPUT_SIZE = 518
    MODEL_PARAMS = {
        "Prompt_length": 12, 
        "learnabel_text_embedding_depth": 9, 
        "learnabel_text_embedding_length": 4
    }
    SIGMA = 4
    
    # 3. 摄像头与硬件
    CAMERA_INDEX = 0       # 摄像头索引，笔记本自带通常是0，外接是1
    CAM_WIDTH = 1280       # 摄像头分辨率宽
    CAM_HEIGHT = 720       # 摄像头分辨率高

    # 4. 机械臂网络配置
    DEFAULT_ROBOT_IP = "192.168.174.59"  # 如果连的是ESP8266热点，通常是这个
    ROBOT_UDP_PORT = 8888             # 对应 .ino 代码中的 UPort

    # 5. 坐标标定 (关键！！需要你拿尺子量)
    # 比例系数 (mm/pixel): 100个像素代表多少毫米？
    SCALE_X = 0.2 #255/1280 
    SCALE_Y = 0.2 #142/720

    # 偏移量 (mm): 摄像头画面中心点 (640, 360) 对应机械臂的坐标是多少？
    OFFSET_X = 137  # 假设中心点对应 X=150mm
    OFFSET_Y = 27   # 假设中心点对应 Y=0mm

    # 6. 动作参数
    Z_SAFE = 0       # 移动时的安全高度
    Z_GRAB = -30     # 抓取高度 (你提到的接地高度)
    BIN_X = 0      # 废料区 X
    BIN_Y = 100      # 废料区 Y

# ================= 机械臂控制器 (UDP 版) =================
class RobotController:
    def __init__(self, target_ip, port=8888):
        self.ip = target_ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(1.0)
        print(f"[Robot] UDP 初始化: 目标 {self.ip}:{self.port}")

    def send_cmd(self, cmd):
        """发送指令到 ESP8266"""
        try:
            msg = cmd.encode('utf-8')
            self.sock.sendto(msg, (self.ip, self.port))
            print(f"[UDP Send] {cmd}")
        except Exception as e:
            print(f"[UDP Error] {e}")

    def execute_pick_and_place(self, x, y):
        """执行完整的分拣流程"""
        # 1. 移动到目标上方 (安全高度)
        self.send_cmd(f"go {x:.1f},{y:.1f},{Config.Z_SAFE}")
        time.sleep(4) # 等待机械臂运动

        # 2. 下探 (抓取高度)
        self.send_cmd(f"go {x:.1f},{y:.1f},{Config.Z_GRAB}")
        time.sleep(1.5)

        # 3. 开泵 (吸取) P -1 为常开
        #self.send_cmd("P 2000")
        #time.sleep(2.5)

        # 4. 抬起
        self.send_cmd(f"go {x:.1f},{y:.1f},{Config.Z_SAFE}")
        time.sleep(1.5)

        # 5. 移动到废料区
        self.send_cmd(f"go {Config.BIN_X},{Config.BIN_Y},{Config.Z_SAFE}")
        time.sleep(4.0)

        # 6. 关泵 (放下) 并 泄气 (O 500)
        #self.send_cmd("O 500")
        time.sleep(0.7)

        # 7. (可选) 回归原点

class RobotActionThread(QThread):
    """独立线程执行机械臂动作，避免阻塞视频显示"""
    action_finished = pyqtSignal()  # 动作完成信号

    def __init__(self, robot, x, y):
        super().__init__()
        self.robot = robot
        self.x = x
        self.y = y

    def run(self):
        try:
            # 执行完整分拣流程
            self.robot.send_cmd(f"go {self.x:.1f},{self.y:.1f},{Config.Z_SAFE}")
            time.sleep(3)
            
            self.robot.send_cmd(f"go {self.x:.1f},{self.y:.1f},{Config.Z_GRAB}")
            time.sleep(1)
            
            # 开泵（如果需要）
            #self.robot.send_cmd("P 3000")
            #time.sleep(3.5)
            
            #self.robot.send_cmd(f"go {self.x:.1f},{self.y:.1f},{Config.Z_SAFE}")
            #time.sleep(1.5)
            
            #self.robot.send_cmd(f"go {Config.BIN_X},{Config.BIN_Y},{Config.Z_SAFE}")
            self.robot.send_cmd(f"go {Config.BIN_X},{Config.BIN_Y},{Config.Z_SAFE}")
            time.sleep(4.0)
            
            # 关泵（如果需要）
            self.robot.send_cmd("O 500")
            #time.sleep(0.7)
        finally:
            self.action_finished.emit()  # 发送动作完成信号

# ================= 核心检测算法 =================
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
        results = self.yolo_model(cv2_img, conf=Config.YOLO_CONF_THRESH, verbose=False)
        detected_boxes = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
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

# ================= 工作线程 (检测+控制) =================
class DetectorThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray) 
    result_signal = pyqtSignal(list) 
    status_signal = pyqtSignal(str) 

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.detector = None
        self.interval = 5 
        self.threshold = 0.65 
        self.camera_index = Config.CAMERA_INDEX
        self.is_detecting = False 
        
        # 机械臂控制器
        self.robot = None
        self.robot_ip = Config.DEFAULT_ROBOT_IP
        self.robot_connected = False
        self.robot_busy = False 

    def set_robot_ip(self, ip):
        self.robot_ip = ip
        try:
            self.robot = RobotController(self.robot_ip, Config.ROBOT_UDP_PORT)
            self.robot_connected = True
            self.status_signal.emit(f"网络目标已设定: {ip}")
        except Exception as e:
            self.status_signal.emit(f"网络配置错误: {e}")

    def pixel_to_world(self, u, v):
        """
        像素 (u,v) -> 机械臂 (x,y)
        请根据实际方向修改这里的加减号
        """
        cx = Config.CAM_WIDTH / 2
        cy = Config.CAM_HEIGHT / 2
        
        # 假设：摄像头上方 对应 机械臂远端 (X+)
        # 假设：摄像头右方 对应 机械臂右侧 (Y- 或 Y+)
        
        # 示例公式 (需要根据实际标定修改符号)
        # v (行) 越小，说明物体越靠上，机械臂X应该越大
        world_x = Config.OFFSET_X - (cx - u) * Config.SCALE_X
        # u (列) 越小，说明物体越靠左，机械臂Y应该越大 (假设左正右负)
        world_y = Config.OFFSET_Y + (cy - v) * Config.SCALE_Y
        
        return world_x, world_y

    def run(self):
        self.status_signal.emit("正在加载 AI 模型...")
        try:
            self.detector = AnomalyDetector()
            self.detector.load_model()
            self.status_signal.emit("模型就绪，请连接机械臂")
        except Exception as e:
            self.status_signal.emit(f"AI 加载失败: {str(e)}")
            return

        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened(): cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAM_HEIGHT)

        last_check_time = time.time()
        cached_results = []
        show_until = 0

        # 添加机械臂动作线程变量
        self.robot_action_thread = None

        while self._run_flag:
            ret, frame = cap.read()
            if not ret: 
                time.sleep(0.1)
                continue
            
            display_frame = frame.copy()
            current_time = time.time()

            # === 检测逻辑 ===
            # 只有在不检测、机械臂不忙、且间隔时间到的情况下才检测
            if self.is_detecting and not self.robot_busy and (current_time - last_check_time >= self.interval):
                self.status_signal.emit("正在扫描...")
                
                boxes = self.detector.detect_objects_yolo(frame)
                
                if len(boxes) > 0:
                    new_results = []
                    ng_target_pixel = None 
                    
                    for box in boxes:
                        x1, y1, x2, y2 = box
                        x1=max(0,x1-10); y1=max(0,y1-10); x2=min(Config.CAM_WIDTH,x2+10); y2=min(Config.CAM_HEIGHT,y2+10)
                        roi = frame[y1:y2, x1:x2]
                        if roi.size == 0: continue
                        
                        try:
                            score, raw_map = self.detector.predict_anomaly(roi)
                            status = "NG" if score > self.threshold else "OK"
                            
                            # 热力图可视化
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
                            
                            # 锁定第一个 NG 目标
                            if status == "NG" and ng_target_pixel is None:
                                cx = (x1 + x2) / 2
                                cy = (y1 + y2) / 2
                                ng_target_pixel = (cx, cy)
                                
                        except Exception as e:
                            print(e)
                    
                    cached_results = new_results
                    # 修改1：热力图只显示1秒
                    show_until = current_time + 2.0  
                    last_check_time = current_time
                    self.result_signal.emit(new_results)
                    
                    # === 触发分拣动作 ===
                    if ng_target_pixel and self.robot_connected:
                        u, v = ng_target_pixel
                        rx, ry = self.pixel_to_world(u, v)
                        
                        self.status_signal.emit(f"执行分拣: 像素({u:.0f},{v:.0f}) -> 机械臂({rx:.1f},{ry:.1f})")

                        # 修改2：使用独立线程执行机械臂动作，避免阻塞画面
                        self.robot_busy = True
                        self.robot_action_thread = RobotActionThread(self.robot, rx, ry)
                        self.robot_action_thread.action_finished.connect(self.on_robot_action_finished)
                        self.robot_action_thread.start()
                        
                else:
                    self.status_signal.emit("视野内无目标")
                    cached_results = []

            # === 绘制逻辑 (保持显示上一帧的结果) ===
            # 只在1秒内显示热力图，之后恢复实时画面
            if current_time < show_until and len(cached_results) > 0:
                for res in cached_results:
                    x1, y1, x2, y2 = res['box']
                    status = res['status']
                    score = res['score']
                    overlay = res['overlay']
                    
                    # 贴回热力图
                    th, tw = overlay.shape[:2]
                    if (y2-y1) == th and (x2-x1) == tw:
                        display_frame[y1:y2, x1:x2] = overlay
                    
                    color = (0, 0, 255) if status == "NG" else (0, 255, 0)
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{status} {score:.2f}"
                    cv2.putText(display_frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # 无论是否检测，都实时更新画面
            self.change_pixmap_signal.emit(display_frame)
            time.sleep(0.03)

        cap.release()

    # 添加机械臂动作完成回调
    def on_robot_action_finished(self):
        self.robot_busy = False
        self.status_signal.emit("分拣完成，继续监控")
        self.last_check_time = time.time()  # 重置检测间隔计时器

    def stop(self):
        self._run_flag = False
        self.wait()

    def update_params(self, threshold, interval):
        self.threshold = threshold
        self.interval = interval

    def toggle_detection(self, active):
        self.is_detecting = active

# ================= 主界面 =================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI 视觉质检 & 机械臂分拣系统 (UDP版)")
        self.resize(1400, 900)
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; color: #ffffff; }
            QLabel { color: #e0e0e0; font-family: 'Microsoft YaHei'; }
            QGroupBox { border: 1px solid #3e3e3e; border-radius: 5px; margin-top: 10px; font-weight: bold; color: #aaaaaa; }
            QPushButton { background-color: #0078d7; color: white; border: none; border-radius: 5px; padding: 10px; font-weight: bold; }
            QPushButton:disabled { background-color: #555; color: #888; }
            QLineEdit { padding: 5px; border-radius: 3px; background-color: #333; color: white; border: 1px solid #555; }
            QListWidget { background-color: #252526; border: 1px solid #3e3e3e; border-radius: 5px; color: #dddddd; }
        """)
        self.init_ui()
        self.start_thread()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)

        # 左侧：视频显示
        video_group = QGroupBox("实时监控")
        video_layout = QVBoxLayout()
        self.image_label = QLabel("正在初始化摄像头...")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #000;")
        self.image_label.setMinimumSize(960, 540)
        video_layout.addWidget(self.image_label)
        video_group.setLayout(video_layout)
        
        # 右侧：控制面板
        control_layout = QVBoxLayout()
        
        # 1. 机械臂连接 (网络版)
        net_group = QGroupBox("机械臂网络连接")
        net_layout = QVBoxLayout()
        
        self.input_ip = QLineEdit(Config.DEFAULT_ROBOT_IP)
        self.input_ip.setPlaceholderText("输入 ESP8266 IP 地址")
        
        self.btn_connect = QPushButton("连接机械臂 (UDP)")
        self.btn_connect.clicked.connect(self.connect_robot)
        
        net_layout.addWidget(QLabel("目标 IP 地址:"))
        net_layout.addWidget(self.input_ip)
        net_layout.addWidget(self.btn_connect)
        net_group.setLayout(net_layout)
        control_layout.addWidget(net_group)

        # 2. 状态显示
        self.lbl_result = QLabel("等待指令")
        self.lbl_result.setStyleSheet("font-size: 20px; font-weight: bold; color: #888; background-color: #2d2d2d; padding: 10px; border-radius: 5px;")
        self.lbl_result.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(self.lbl_result)

        # 3. 参数设置
        settings_group = QGroupBox("算法参数")
        s_layout = QVBoxLayout()
        
        self.lbl_thresh = QLabel("判定阈值: 0.65")
        self.slider_thresh = QSlider(Qt.Horizontal)
        self.slider_thresh.setRange(0, 100)
        self.slider_thresh.setValue(65)
        self.slider_thresh.valueChanged.connect(self.on_param_change)
        
        self.lbl_interval = QLabel("检测间隔: 5秒")
        self.slider_interval = QSlider(Qt.Horizontal)
        self.slider_interval.setRange(2, 30)
        self.slider_interval.setValue(5)
        self.slider_interval.valueChanged.connect(self.on_param_change)
        
        s_layout.addWidget(self.lbl_thresh)
        s_layout.addWidget(self.slider_thresh)
        s_layout.addWidget(self.lbl_interval)
        s_layout.addWidget(self.slider_interval)
        settings_group.setLayout(s_layout)
        control_layout.addWidget(settings_group)

        # 4. 开始按钮
        self.btn_start = QPushButton("开始自动分拣")
        self.btn_start.setCheckable(True)
        self.btn_start.clicked.connect(self.toggle_detection)
        self.btn_start.setEnabled(False) # 模型加载完才可用
        control_layout.addWidget(self.btn_start)

        # 5. 日志
        log_group = QGroupBox("运行日志")
        l_layout = QVBoxLayout()
        self.list_log = QListWidget()
        l_layout.addWidget(self.list_log)
        log_group.setLayout(l_layout)
        control_layout.addWidget(log_group)
        
        self.lbl_status = QLabel("系统初始化中...")
        control_layout.addWidget(self.lbl_status)

        main_layout.addWidget(video_group, stretch=3)
        main_layout.addLayout(control_layout, stretch=1)

    def start_thread(self):
        self.thread = DetectorThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.result_signal.connect(self.update_log)
        self.thread.status_signal.connect(self.update_status)
        self.thread.start()

    def connect_robot(self):
        ip = self.input_ip.text()
        if not ip:
            QMessageBox.warning(self, "错误", "请输入 IP 地址")
            return
        self.thread.set_robot_ip(ip)
        self.btn_connect.setText(f"已设定目标: {ip}")
        self.btn_connect.setStyleSheet("background-color: #388e3c;")
        # 解锁开始按钮 (如果模型也好了的话)
        if self.btn_start.text() != "开始自动分拣": 
             pass # 已经在运行

    def update_image(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        convert_to_Qt_format = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio)
        self.image_label.setPixmap(QPixmap.fromImage(p))

    def update_log(self, results_list):
        timestamp = datetime.now().strftime("%H:%M:%S")
        ng_items = [r for r in results_list if r['status'] == 'NG']
        
        if len(ng_items) > 0:
            self.lbl_result.setText(f"发现 {len(ng_items)} 个 NG")
            self.lbl_result.setStyleSheet("font-size: 24px; background-color: #d32f2f; color: white;")
            for item in ng_items:
                log_txt = f"[{timestamp}] NG | Score: {item['score']:.3f}"
                list_item = QListWidgetItem(log_txt)
                list_item.setForeground(QColor("#ff5555"))
                self.list_log.insertItem(0, list_item)
        else:
            self.lbl_result.setText(f"OK ({len(results_list)})")
            self.lbl_result.setStyleSheet("font-size: 24px; background-color: #388e3c; color: white;")
            list_item = QListWidgetItem(f"[{timestamp}] All OK")
            list_item.setForeground(QColor("#55ff55"))
            self.list_log.insertItem(0, list_item)

    def update_status(self, text):
        self.lbl_status.setText(text)
        if "模型就绪" in text:
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
            if not self.thread.robot_connected:
                ret = QMessageBox.question(self, "警告", "机械臂未连接，只进行视觉检测？", QMessageBox.Yes | QMessageBox.No)
                if ret == QMessageBox.No:
                    self.btn_start.setChecked(False)
                    return
            
            self.btn_start.setText("停止分拣")
            self.btn_start.setStyleSheet("background-color: #d32f2f;")
            self.thread.toggle_detection(True)
        else:
            self.btn_start.setText("开始自动分拣")
            self.btn_start.setStyleSheet("background-color: #0078d7;")
            self.thread.toggle_detection(False)
            self.lbl_result.setText("等待指令")
            self.lbl_result.setStyleSheet("background-color: #2d2d2d; color: #888;")

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())