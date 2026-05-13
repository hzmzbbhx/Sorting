import sys
import cv2
import time
import socket
import threading
import torch
import numpy as np
import math
from datetime import datetime
from PIL import Image

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, 
                             QVBoxLayout, QHBoxLayout, QPushButton, QSlider, 
                             QGroupBox, QListWidget, QListWidgetItem, QLineEdit, QMessageBox, QCheckBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QColor

# === 核心算法依赖 ===
try:
    from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode
    from peft import PeftModel
    from scipy.ndimage import gaussian_filter
    import AnomalyCLIP_lib
    from prompt_ensemble import AnomalyCLIP_PromptLearner
    from AnomalyCLIP_lib.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
    from ultralytics import YOLO
except ImportError as e:
    print(f"依赖缺失: {e}")

# === 工具函数: 稳健归一化 (本地定义，修复全屏变色问题) ===
def normalize(pred, max_value=None, min_value=None):
    # 强制转为 float32 防止数据类型溢出
    pred = np.array(pred, dtype=np.float32)
    
    if max_value is None or min_value is None:
        min_v, max_v = pred.min(), pred.max()
        # 关键修复：如果图像完全一样（全黑或全白），返回全0，防止除以0产生NaN导致变色
        if max_v == min_v:
            return np.zeros_like(pred)
        return (pred - min_v) / (max_v - min_v)
    else:
        if max_value == min_value:
            return np.zeros_like(pred)
        return (pred - min_value) / (max_value - min_value)

# ================= 配置区域 =================
class Config:
    # --- 模型路径 ---
    CHECKPOINT_PATH = './checkpoint_lora_pokerback/best_model.pth'
    LORA_PATH = './checkpoint_lora_pokerback/best_lora'
    BASE_MODEL = "ViT-L/14@336px"
    YOLO_MODEL_PATH = 'D:/test/AnomalyCLIP/runs/detect/train3/weights/best.pt' 
    YOLO_CONF_THRESH = 0.5
    
    # --- 模型参数 ---
    INPUT_SIZE = 518
    MODEL_PARAMS = {
        "Prompt_length": 12, 
        "learnabel_text_embedding_depth": 9, 
        "learnabel_text_embedding_length": 4
    }
    SIGMA = 4
    
    # --- 机械臂物理参数 (new2.py 参数) ---
    L1_BASE_H = 15.0    
    L2_ARM = 12.0       
    L3_FOREARM = 12.0   
    L4_HAND_H = 5.5     
    L4_HAND_V = 4.0     

    # 脉冲比例
    RATIO_J1 = 41.0    
    RATIO_J2 = 37.0   
    RATIO_J3 = 37.0   

    # Home点 (L型)
    HOME_ANGLES = [0.0, 90.0, 0.0] 

    # --- 摄像头与标定 ---
    CAMERA_INDEX = 0       
    CAM_WIDTH = 1280       
    CAM_HEIGHT = 720       

    SCALE_X = 0.03   
    SCALE_Y = 0.03   
    OFFSET_X = 17.5  
    OFFSET_Y = 0     

    # --- 动作参数 ---
    Z_SAFE = 2.0     # 安全高度 (抬起高度)
    Z_GRAB = -0.1    # 抓取高度
    
    # 废料区 (XYZ)
    BIN_X = 8.0      
    BIN_Y = -15.7      
    BIN_Z = 7.1

    # 速度控制
    DEFAULT_SPEED_US = 400   # 默认速度
    START_SPEED_US = 800    # 起步速度    
    INTERPOLATION_STEP = 2.0 

    # 网络
    DEFAULT_ROBOT_IP = "192.168.1.12" 
    ROBOT_UDP_PORT = 7788

# ================= 机械臂控制器 (S-Curve) =================
class RobotController:
    def __init__(self, target_ip, port=7788):
        self.ip = target_ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(0.5)
        
        self.current_angles = list(Config.HOME_ANGLES) 
        self.last_steps = [0, 0, 0] 
        self.current_speed_us = Config.DEFAULT_SPEED_US
        
        print(f"[Robot] 初始化: {self.ip}:{self.port}")
        self.send_raw_cmd(0, 0) # Enable
        self.send_raw_cmd(4, Config.START_SPEED_US) 

    def set_speed(self, delay_us):
        self.current_speed_us = int(delay_us)
        # 实时同步速度给ESP32
        self.send_raw_cmd(4, self.current_speed_us)

    def close(self):
        try:
            self.send_raw_cmd(0, 1) # Disable
            self.sock.close()
            print("[Robot] 连接断开")
        except: pass

    def send_raw_cmd(self, cmd, val):
        try:
            msg = f"{cmd},{val}".encode('utf-8')
            self.sock.sendto(msg, (self.ip, self.port))
            time.sleep(0.002)
        except: pass

    def control_tool(self, action):
        if action == "suck":
            self.send_raw_cmd(5, 1) # 吸气
        elif action == "release":
            self.send_raw_cmd(5, 3) # 释放
        elif action == "stop":
            self.send_raw_cmd(5, 2) # 停止

    def solve_ik(self, x, y, z):
        # 几何解算
        t1 = math.degrees(math.atan2(y, x))
        r_end = math.sqrt(x**2 + y**2)
        r_wrist = r_end - Config.L4_HAND_H
        z_wrist = z + Config.L4_HAND_V 
        dz_arm = z_wrist - Config.L1_BASE_H
        dr_arm = r_wrist 
        hyp = math.sqrt(dr_arm**2 + dz_arm**2)
        
        if hyp > (Config.L2_ARM + Config.L3_FOREARM) or hyp < abs(Config.L2_ARM - Config.L3_FOREARM):
            # print(f"[IK] 不可达: Hyp={hyp:.1f}")
            return None
            
        try:
            cos_alpha = (Config.L2_ARM**2 + hyp**2 - Config.L3_FOREARM**2) / (2 * Config.L2_ARM * hyp)
            alpha = math.acos(cos_alpha)
            beta = math.atan2(dz_arm, dr_arm)
            t2 = math.degrees(beta + alpha) 
            
            r_elbow = Config.L2_ARM * math.cos(math.radians(t2))
            z_elbow = Config.L2_ARM * math.sin(math.radians(t2))
            r_vec_l3 = dr_arm - r_elbow
            z_vec_l3 = dz_arm - z_elbow
            t3_abs = math.degrees(math.atan2(z_vec_l3, r_vec_l3))
            
            return [t1, t2, t3_abs]
        except: return None

    def move_to(self, x, y, z):
        target_angles = self.solve_ik(x, y, z)
        if not target_angles: return False
        self.smooth_move_to_angles(target_angles)
        return True

    def smooth_move_to_angles(self, target_angles):
        """S-Curve 插补"""
        start_angles = self.current_angles
        diffs = [abs(t - s) for t, s in zip(target_angles, start_angles)]
        max_diff = max(diffs)
        
        if max_diff < 0.5: return

        steps = int(max_diff / Config.INTERPOLATION_STEP)
        if steps < 1: steps = 1
        
        accel_ratio = 0.25
        target_spd = self.current_speed_us 
        start_spd = max(target_spd, Config.START_SPEED_US)

        for i in range(1, steps + 1):
            ratio = i / steps
            
            curr_delay = target_spd
            if steps > 3:
                if ratio <= accel_ratio: 
                    p = ratio / accel_ratio
                    curr_delay = start_spd - (start_spd - target_spd) * p
                elif ratio >= (1 - accel_ratio):
                    p = (ratio - (1 - accel_ratio)) / accel_ratio
                    curr_delay = target_spd + (start_spd - target_spd) * p
            curr_delay = int(max(200, curr_delay))

            cur_angles = [s + (t - s) * ratio for s, t in zip(start_angles, target_angles)]
            
            self.send_raw_cmd(4, curr_delay)
            self.send_motor_commands(cur_angles, curr_delay)
            
        self.current_angles = target_angles

    def send_motor_commands(self, angles, delay_us):
        d1 = (angles[0] - Config.HOME_ANGLES[0]) * Config.RATIO_J1
        d2 = (angles[1] - Config.HOME_ANGLES[1]) * Config.RATIO_J2
        d3 = (angles[2] - Config.HOME_ANGLES[2]) * Config.RATIO_J3
        target_steps = [int(d1), int(d2), int(d3)]
        
        inc = [t - l for t, l in zip(target_steps, self.last_steps)]
        
        self.send_axis_blocking(1, inc[0], delay_us)
        self.send_axis_blocking(3, inc[2], delay_us)
        self.send_axis_blocking(2, inc[1], delay_us)
            
        self.last_steps = target_steps

    def send_axis_blocking(self, axis, steps, delay_us):
        if steps == 0: return
        self.send_raw_cmd(axis, steps)
        # 精确延时防丢包
        wait = abs(steps) * delay_us * 2 / 1000000.0
        time.sleep(wait + 0.002)

# ================= 动作执行线程 (逻辑修正版) =================
class RobotActionThread(QThread):
    action_finished = pyqtSignal()
    log_signal = pyqtSignal(str)

    def __init__(self, robot, x, y, mode="pick"):
        super().__init__()
        self.robot = robot
        self.tx, self.ty = x, y
        self.mode = mode 
        self._stop = False

    def stop(self): self._stop = True

    def run(self):
        r = self.robot
        try:
            if self._stop: return
            
            if self.mode == "move_bin":
                self.log_signal.emit(f"-> 避让: 废料区 ({Config.BIN_X}, {Config.BIN_Y})")
                r.move_to(Config.BIN_X, Config.BIN_Y, Config.Z_SAFE)
                return

            # --- 优化后的分拣流程 ---
            # 1. 移动到目标上方的安全高度
            self.log_signal.emit(f"-> 移动到上方 ({self.tx:.1f}, {self.ty:.1f}, Z={Config.Z_SAFE})")
            if not r.move_to(self.tx, self.ty, Config.Z_SAFE):
                self.log_signal.emit("目标不可达 (超出范围)")
                return # 退出，不再继续
            
            # 2. 垂直下探抓取
            self.log_signal.emit(f"-> 下探抓取 (Z={Config.Z_GRAB})")
            r.move_to(self.tx, self.ty, Config.Z_GRAB)
            
            # 吸取
            self.log_signal.emit("-> 吸气")
            r.control_tool("suck")
            time.sleep(1.0) # 等待真空建立
            r.control_tool("stop")
            if self._stop: return

            # 3. 垂直抬起回安全高度 (防止斜向拖拽)
            self.log_signal.emit("-> 垂直抬起")
            r.move_to(self.tx, self.ty, Config.Z_SAFE)

            # 4. 平移到废料区上方
            self.log_signal.emit(f"-> 移至废料区 ({Config.BIN_X}, {Config.BIN_Y})")
            r.move_to(Config.BIN_X, Config.BIN_Y, Config.Z_SAFE)
            
            # 5. 释放
            self.log_signal.emit("-> 释放")
            r.control_tool("release")
            
            time.sleep(0.6)
            
            self.log_signal.emit("-> 完成 (原地待命)")

        except Exception as e:
            self.log_signal.emit(f"动作异常: {e}")
        finally:
            self.action_finished.emit()

# ================= 真实检测类 =================
class AnomalyDetector:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.preprocess = None
        self.text_features = None
        self.prompt_learner = None
        self.yolo_model = None

    def load_model(self):
        print(f"[Init] Loading YOLO...")
        self.yolo_model = YOLO(Config.YOLO_MODEL_PATH)
        
        print(f"[Init] Loading AnomalyCLIP...")
        self.model, _ = AnomalyCLIP_lib.load(Config.BASE_MODEL, device=self.device, design_details=Config.MODEL_PARAMS)
        self.model.visual.DAPM_replace(DPAM_layer=20)
        self.model.visual = PeftModel.from_pretrained(self.model.visual, Config.LORA_PATH)
        self.model.eval(); self.model.to(self.device)
        
        self.prompt_learner = AnomalyCLIP_PromptLearner(self.model, Config.MODEL_PARAMS)
        ckpt = torch.load(Config.CHECKPOINT_PATH, map_location='cpu')
        state_dict = ckpt["prompt_learner"] if "prompt_learner" in ckpt else ckpt
        self.prompt_learner.load_state_dict(state_dict)
        self.prompt_learner.to(self.device); self.prompt_learner.eval()
        
        with torch.no_grad():
            prompts, toks, comp_prompts = self.prompt_learner(cls_id=None)
            text_feats = self.model.encode_text_learn(prompts, toks, comp_prompts).float()
            text_feats = torch.stack(torch.chunk(text_feats, dim=0, chunks=2), dim=1)
            self.text_features = text_feats / text_feats.norm(dim=-1, keepdim=True)
            
        self.preprocess = Compose([
            Resize((Config.INPUT_SIZE, Config.INPUT_SIZE), interpolation=InterpolationMode.BICUBIC),
            ToTensor(),
            Normalize(OPENAI_DATASET_MEAN, OPENAI_DATASET_STD)
        ])
        print("[Init] Ready.")

    def detect_objects_yolo(self, img):
        res = self.yolo_model(img, conf=Config.YOLO_CONF_THRESH, verbose=False)
        return [box.xyxy[0].cpu().numpy().astype(int) for r in res for box in r.boxes]

    def predict_anomaly(self, crop_img):
        img_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        img_ts = self.preprocess(Image.fromarray(img_rgb)).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            img_feat, patch_feat = self.model.encode_image(img_ts, feature_list=[6, 12, 18, 24], DPAM_layer=20)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            
            score = (img_feat.unsqueeze(1) @ self.text_features.permute(0,2,1) / 0.07).softmax(-1)[:, 0, 1].item()
            
            sim_maps = []
            for pf in patch_feat:
                pf = pf / pf.norm(dim=-1, keepdim=True)
                sim, _ = AnomalyCLIP_lib.compute_similarity(pf, self.text_features[0])
                sim_maps.append(AnomalyCLIP_lib.get_similarity_map(sim[:, 1:, :], Config.INPUT_SIZE))
            
            raw_map = torch.stack(sim_maps, dim=0).mean(dim=0)[0, :, :, 1].cpu().numpy()
            return score, gaussian_filter(raw_map, sigma=Config.SIGMA)

# ================= 视觉线程 =================
class DetectorThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray) 
    result_signal = pyqtSignal(list) 
    status_signal = pyqtSignal(str) 

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.detector = None
        self.threshold = 0.5 
        self.interval = 5 
        self.is_detecting = False 
        self.robot_enabled = True # 默认开启
        self.show_heatmap = True   
        self.robot = None
        self.robot_busy = False 
        self.action_thread = None

    def connect_robot(self, ip):
        try:
            self.robot = RobotController(ip, Config.ROBOT_UDP_PORT)
            self.status_signal.emit(f"已连接: {ip}")
            self.trigger_move_bin()
            return True
        except Exception as e:
            self.status_signal.emit(f"连接失败: {e}")
            return False

    def disconnect_robot(self):
        if self.robot:
            if self.action_thread: self.action_thread.stop()
            self.robot.close()
            self.robot = None
            self.robot_busy = False
            self.status_signal.emit("已断开")
            return True
        return False

    def trigger_move_bin(self):
        if self.robot:
            self.robot_busy = True
            self.action_thread = RobotActionThread(self.robot, 0, 0, mode="move_bin")
            self.action_thread.log_signal.connect(self.status_signal.emit)
            self.action_thread.action_finished.connect(self.on_finished)
            self.action_thread.start()

    def update_robot_speed(self, val):
        if self.robot:
            self.robot.set_speed(val)

    def pixel_to_world(self, u, v):
        cx, cy = Config.CAM_WIDTH/2, Config.CAM_HEIGHT/2
        return Config.OFFSET_X - (cy-v)*Config.SCALE_X, Config.OFFSET_Y - (u-cx)*Config.SCALE_Y

    def run(self):
        try:
            self.detector = AnomalyDetector() 
            self.detector.load_model()
            self.status_signal.emit("模型加载完成")
        except Exception as e:
            self.status_signal.emit(f"模型失败: {e}"); return

        cap = cv2.VideoCapture(Config.CAMERA_INDEX)
        cap.set(3, Config.CAM_WIDTH); cap.set(4, Config.CAM_HEIGHT)
        last_check = time.time(); cached_res = []; show_until = 0

        while self._run_flag:
            ret, frame = cap.read()
            if not ret: time.sleep(0.1); continue
            disp = frame.copy()
            now = time.time()

            if self.is_detecting and not self.robot_busy and (now - last_check >= self.interval):
                self.status_signal.emit("扫描中...")
                boxes = self.detector.detect_objects_yolo(frame)
                
                if boxes:
                    res_list = []; target = None
                    for b in boxes:
                        x1, y1, x2, y2 = b
                        pad = 10
                        x1=max(0,x1-pad); y1=max(0,y1-pad); x2=min(1280,x2+pad); y2=min(720,y2+pad)
                        roi = frame[y1:y2, x1:x2]
                        if roi.size == 0: continue
                        
                        try:
                            score, rmap = self.detector.predict_anomaly(roi)
                            stat = "NG" if score > self.threshold else "OK"
                            
                            overlay = None
                            if self.show_heatmap:
                                rmap_res = cv2.resize(rmap, (roi.shape[1], roi.shape[0]))
                                norm_map = normalize(rmap_res) # 使用本地的 robust normalize
                                hmap = cv2.applyColorMap((norm_map*255).astype(np.uint8), cv2.COLORMAP_JET)
                                overlay = cv2.addWeighted(roi, 0.6, hmap, 0.4, 0)

                            res_list.append({'box':b, 'st':stat, 'sc':score, 'ov':overlay})
                            if stat == "NG" and target is None: target = ((x1+x2)/2, (y1+y2)/2)
                        except: pass
                    
                    cached_res = res_list; show_until = now + 2.0; last_check = now
                    self.result_signal.emit(res_list)
                    
                    if target and self.robot and self.robot_enabled:
                        rx, ry = self.pixel_to_world(*target)
                        
                        # 先检查IK是否可达，防止发无效指令
                        if self.robot.solve_ik(rx, ry, Config.Z_SAFE):
                            self.status_signal.emit(f"执行抓取: ({rx:.1f}, {ry:.1f})")
                            self.robot_busy = True
                            self.action_thread = RobotActionThread(self.robot, rx, ry)
                            self.action_thread.log_signal.connect(self.status_signal.emit)
                            self.action_thread.action_finished.connect(self.on_finished)
                            self.action_thread.start()
                        else:
                            self.status_signal.emit(f"目标超限: ({rx:.1f}, {ry:.1f})")
                    elif target and not self.robot_enabled:
                        self.status_signal.emit("发现异常 (抓取已禁用)")
                        
                else:
                    self.status_signal.emit("无目标")

            # 绘制
            if now < show_until and cached_res:
                for r in cached_res:
                    x1,y1,x2,y2 = r['box']
                    # 热力图绘制
                    if self.show_heatmap and r['ov'] is not None:
                        h,w = r['ov'].shape[:2]
                        if y2-y1 == h and x2-x1 == w: disp[y1:y2, x1:x2] = r['ov']
                    
                    col = (0,0,255) if r['st']=="NG" else (0,255,0)
                    cv2.rectangle(disp, (x1,y1), (x2,y2), col, 2)
                    cv2.putText(disp, f"{r['st']} {r['sc']:.2f}", (x1,y1-5), 0, 0.6, col, 2)

            self.change_pixmap_signal.emit(disp)
            time.sleep(0.03)
        cap.release()

    def on_finished(self):
        self.robot_busy = False
        self.last_check = time.time()
        self.status_signal.emit("动作完成")

    def stop(self): self._run_flag = False; self.wait()
    def update_cfg(self, t, i, re, sh): self.threshold=t; self.interval=i; self.robot_enabled=re; self.show_heatmap=sh
    def toggle(self, v): 
        self.is_detecting = v
        if v: self.robot_busy = False

# ================= 界面 =================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(1200, 800)
        w = QWidget(); self.setCentralWidget(w); ly = QHBoxLayout(w)
        
        # 视频区
        v_box = QGroupBox("监控"); ly.addWidget(v_box, 3)
        self.lbl_v = QLabel("初始化..."); self.lbl_v.setAlignment(Qt.AlignCenter); self.lbl_v.setStyleSheet("background:black;")
        v_ly = QVBoxLayout(); v_ly.addWidget(self.lbl_v); v_box.setLayout(v_ly)
        
        # 控制区
        c_ly = QVBoxLayout(); ly.addLayout(c_ly, 1)
        self.lbl_s = QLabel("就绪"); c_ly.addWidget(self.lbl_s)
        
        # 连接
        net = QHBoxLayout()
        self.ip = QLineEdit(Config.DEFAULT_ROBOT_IP)
        b_con = QPushButton("连接"); b_con.clicked.connect(self.con)
        b_dis = QPushButton("急停"); b_dis.clicked.connect(self.dis); b_dis.setStyleSheet("background:#d32f2f")
        net.addWidget(self.ip); net.addWidget(b_con); net.addWidget(b_dis)
        c_ly.addLayout(net)
        
        # 调速
        self.sl_spd = QSlider(Qt.Horizontal); self.sl_spd.setRange(200, 2000); self.sl_spd.setValue(600)
        self.sl_spd.setInvertedAppearance(False) 
        self.sl_spd.valueChanged.connect(self.upd_spd)
        self.lb_spd = QLabel("速度(延迟): 400us")
        c_ly.addWidget(self.lb_spd); c_ly.addWidget(self.sl_spd)

        # 参数
        self.sl_t = QSlider(Qt.Horizontal); self.sl_t.setRange(0,100); self.sl_t.setValue(50); self.sl_t.valueChanged.connect(self.upd)
        self.lb_t = QLabel("0.50")
        self.sl_i = QSlider(Qt.Horizontal); self.sl_i.setRange(1,20); self.sl_i.setValue(5); self.sl_i.valueChanged.connect(self.upd)
        self.lb_i = QLabel("5s")
        c_ly.addWidget(QLabel("阈值:")); h1=QHBoxLayout(); h1.addWidget(self.sl_t); h1.addWidget(self.lb_t); c_ly.addLayout(h1)
        c_ly.addWidget(QLabel("间隔:")); h2=QHBoxLayout(); h2.addWidget(self.sl_i); h2.addWidget(self.lb_i); c_ly.addLayout(h2)
        
        # 开关
        self.ck_en = QCheckBox("允许抓取"); self.ck_en.setChecked(True); self.ck_en.stateChanged.connect(self.upd)
        self.ck_hm = QCheckBox("显示热力图"); self.ck_hm.setChecked(True); self.ck_hm.stateChanged.connect(self.upd)
        c_ly.addWidget(self.ck_en); c_ly.addWidget(self.ck_hm)
        
        # 按钮
        h_btn = QHBoxLayout()
        self.b_run = QPushButton("开始运行"); self.b_run.setCheckable(True); self.b_run.clicked.connect(self.run_logic); self.b_run.setEnabled(False)
        self.b_bin = QPushButton("去废料区"); self.b_bin.clicked.connect(self.go_bin); self.b_bin.setEnabled(False)
        h_btn.addWidget(self.b_run); h_btn.addWidget(self.b_bin); c_ly.addLayout(h_btn)
        
        self.log = QListWidget(); c_ly.addWidget(self.log)
        
        self.th = DetectorThread()
        self.th.change_pixmap_signal.connect(self.set_img)
        self.th.status_signal.connect(self.set_st)
        self.th.start()
        
        self.upd() # 初始同步

    def con(self): 
        if self.th.connect_robot(self.ip.text()): 
            self.b_bin.setEnabled(True)
    def dis(self): self.th.disconnect_robot()
    def go_bin(self): self.th.trigger_move_bin()
    def run_logic(self): 
        self.th.toggle(self.b_run.isChecked())
        self.b_run.setText("停止" if self.b_run.isChecked() else "开始运行")
    def set_img(self, img): 
        h,w,c = img.shape
        self.lbl_v.setPixmap(QPixmap.fromImage(QImage(img.data, w, h, c*w, QImage.Format_RGB888)).scaled(self.lbl_v.size(), Qt.KeepAspectRatio))
    def set_st(self, t): 
        self.lbl_s.setText(t); self.log.insertItem(0, f"[{datetime.now().strftime('%H:%M:%S')}] {t}")
        if "模型加载完成" in t: self.b_run.setEnabled(True)
    def upd(self):
        t, i = self.sl_t.value()/100, self.sl_i.value()
        self.lb_t.setText(f"{t:.2f}"); self.lb_i.setText(f"{i}s")
        self.th.update_cfg(t, i, self.ck_en.isChecked(), self.ck_hm.isChecked())
    def upd_spd(self):
        val = self.sl_spd.value()
        self.lb_spd.setText(f"速度(延迟): {val}us (越小越快)")
        self.th.update_robot_speed(val)
    def closeEvent(self, e): self.th.stop()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())