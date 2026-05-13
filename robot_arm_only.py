import sys
import socket
import math
import time
import numpy as np

# PyQt5 Imports
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QTabWidget, 
                             QSlider, QLabel, QDoubleSpinBox, QTextEdit, 
                             QGroupBox, QGridLayout, QVBoxLayout, QHBoxLayout, QPushButton, QMessageBox)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# ================= 配置区域 (已修正为 new2.py 验证参数) =================
# 机械臂物理尺寸 (cm)
L1_BASE_H = 15.0    
L2_ARM = 12.0       
L3_FOREARM = 12.0   
L4_HAND_H = 5.5     
L4_HAND_V = 4.0     

# 脉冲比例 (步/度)
RATIO_J1 = 41.0    
RATIO_J2 = 37.0   
RATIO_J3 = 37.0   

# 初始状态
HOME_THETA_1 = 0.0
HOME_THETA_2 = 90.0  
HOME_THETA_3 = 0.0   

# 初始坐标 (基于 L1=15 计算调整)
HOME_X = 17.5
HOME_Y = 0.0
HOME_Z = 23.0

DEFAULT_IP = "192.168.1.12"
DEFAULT_PORT = 7788

# 插补与运动优化配置
INTERPOLATION_STEP = 2.0   # 恢复为 2.0 度，保证精度
SAFETY_BUFFER_SEC = 0.002  # 安全缓冲
DEFAULT_DELAY_US = 600     # 默认巡航速度
START_DELAY_US = 1200      # 起步速度

# ================= 绘图组件 =================
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111, projection='3d')
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial', 'sans-serif'] 
        plt.rcParams['axes.unicode_minus'] = False
        super(MplCanvas, self).__init__(self.fig)

# ================= 后台运动线程 (S型加减速 + 阻塞发送) =================
class MovementWorker(QThread):
    """
    负责 S-Curve 加减速规划与指令发送的后台线程
    """
    log_signal = pyqtSignal(str)     # 发送日志信号
    finished_signal = pyqtSignal()   # 运动结束信号

    def __init__(self, controller, target_angles):
        super().__init__()
        self.c = controller  # 持有主控制器引用
        self.target_angles = target_angles

    def run(self):
        target_t1, target_t2, target_t3 = self.target_angles
        start_t1, start_t2, start_t3 = self.c.real_angles
        
        # 计算总偏差
        diffs = [abs(target_t1 - start_t1), abs(target_t2 - start_t2), abs(target_t3 - start_t3)]
        max_diff = max(diffs)
        
        # 1. 极小距离移动：直接执行
        if max_diff < 0.5:
            self.send_speed_cmd(self.c.current_speed_us)
            self.send_motor_commands(target_t1, target_t2, target_t3, self.c.current_speed_us)
            self.finished_signal.emit()
            return

        # 2. 轨迹规划
        steps = int(max_diff / INTERPOLATION_STEP)
        if steps < 1: steps = 1
        
        target_speed_us = self.c.current_speed_us
        start_speed_us = max(target_speed_us, START_DELAY_US)
        
        self.log_signal.emit(f"S-Curve轨迹: 距离{max_diff:.1f}°，分{steps}段")
        
        # 3. 循环执行插补
        accel_ratio = 0.25 
        
        for i in range(1, steps + 1):
            ratio = i / steps
            
            # === S-Curve 速度规划 ===
            current_delay = target_speed_us
            if steps > 3:
                if ratio <= accel_ratio:
                    p = ratio / accel_ratio
                    current_delay = start_speed_us - (start_speed_us - target_speed_us) * p
                elif ratio >= (1 - accel_ratio):
                    p = (ratio - (1 - accel_ratio)) / accel_ratio
                    current_delay = target_speed_us + (start_speed_us - target_speed_us) * p
            
            current_delay = int(max(200, current_delay))

            # 计算本段目标角度
            cur_t1 = start_t1 + (target_t1 - start_t1) * ratio
            cur_t2 = start_t2 + (target_t2 - start_t2) * ratio
            cur_t3 = start_t3 + (target_t3 - start_t3) * ratio
            
            # 发送变速指令
            self.send_speed_cmd(current_delay)
            
            # 发送运动指令 (阻塞等待)
            self.send_motor_commands(cur_t1, cur_t2, cur_t3, current_delay)

        self.finished_signal.emit()

    def send_speed_cmd(self, delay_val):
        try:
            msg = f"4,{delay_val}"
            self.c.udp_socket.sendto(msg.encode(), (self.c.ip_input.text(), int(self.c.port_input.text())))
            time.sleep(0.001) 
        except:
            pass

    def send_motor_commands(self, t1, t2, t3, current_delay_us):
        """计算脉冲 -> 逐轴发送并等待 -> 防止ESP32丢包"""
        # 1. 计算目标总脉冲
        d1 = (t1 - HOME_THETA_1) * RATIO_J1
        d2 = (t2 - HOME_THETA_2) * RATIO_J2
        d3 = (t3 - HOME_THETA_3) * RATIO_J3
        target_steps = [int(d1), int(d2), int(d3)]
        
        # 2. 计算本次增量
        inc = [t - l for t, l in zip(target_steps, self.c.last_steps)]
        
        # 3. 逐个发送并等待 (关键修正：必须发一个等一个)
        # 顺序：底座 -> 末端 -> 大臂
        self.send_axis_blocking(1, inc[0], current_delay_us)
        self.send_axis_blocking(3, inc[2], current_delay_us)
        self.send_axis_blocking(2, inc[1], current_delay_us)
            
        # 更新状态
        self.c.last_steps = target_steps
        self.c.real_angles = [t1, t2, t3]

    def send_axis_blocking(self, axis_idx, steps_delta, delay_us):
        if steps_delta == 0: return
        
        msg = f"{axis_idx},{steps_delta}"
        try:
            self.c.udp_socket.sendto(msg.encode(), (self.c.ip_input.text(), int(self.c.port_input.text())))
        except:
            pass
        
        # 动态计算耗时并等待
        move_time_sec = abs(steps_delta) * delay_us * 2 / 1000000.0
        wait_time = move_time_sec + SAFETY_BUFFER_SEC
        time.sleep(wait_time)

# ================= 主控制器 =================
class RobotController(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("D1机械臂控制系统 - 最终修正版")
        self.resize(1150, 750)
        
        # 核心数据
        self.lock_status = False
        self.preview_angles = [HOME_THETA_1, HOME_THETA_2, HOME_THETA_3] 
        self.real_angles = [HOME_THETA_1, HOME_THETA_2, HOME_THETA_3]    
        self.last_steps = [0, 0, 0] 
        self.current_speed_us = DEFAULT_DELAY_US 
        
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # 线程句柄
        self.worker = None
        
        self.init_ui()
        self.update_plot()
        self.log(f"参数已同步: L1={L1_BASE_H}, Ratios={RATIO_J1}/{RATIO_J2}/{RATIO_J3}")

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # --- 左侧：3D 仿真图 ---
        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        main_layout.addWidget(self.canvas, stretch=6)
        
        # --- 右侧：控制区域 ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        main_layout.addWidget(right_panel, stretch=4)
        
        # 1. 连接设置
        grp_conn = QGroupBox("1. 硬件连接与巡航速度")
        layout_conn = QGridLayout()
        self.ip_input = QtWidgets.QLineEdit(DEFAULT_IP)
        self.port_input = QtWidgets.QLineEdit(str(DEFAULT_PORT))
        
        # 速度设置
        self.spin_speed = QtWidgets.QSpinBox()
        self.spin_speed.setRange(100, 3000) 
        self.spin_speed.setSingleStep(50)
        self.spin_speed.setValue(DEFAULT_DELAY_US)
        self.spin_speed.setSuffix(" us")
        self.spin_speed.valueChanged.connect(self.update_speed_var)
        
        self.btn_lock = QPushButton("锁定控制 (Enable)")
        self.btn_lock.setCheckable(True)
        self.btn_lock.clicked.connect(self.toggle_lock)
        self.btn_lock.setStyleSheet("background-color: #ffcccc; font-weight: bold;")
        
        layout_conn.addWidget(QLabel("IP:"), 0, 0)
        layout_conn.addWidget(self.ip_input, 0, 1)
        layout_conn.addWidget(QLabel("Port:"), 0, 2)
        layout_conn.addWidget(self.port_input, 0, 3)
        layout_conn.addWidget(QLabel("MaxSpeed:"), 1, 0)
        layout_conn.addWidget(self.spin_speed, 1, 1)
        layout_conn.addWidget(self.btn_lock, 2, 0, 1, 4)
        grp_conn.setLayout(layout_conn)
        right_layout.addWidget(grp_conn)

        # 2. 分页控制
        self.tabs = QTabWidget()
        right_layout.addWidget(self.tabs)
        
        self.tab_joint = QWidget()
        self.setup_joint_tab()
        self.tabs.addTab(self.tab_joint, "关节角度 (FK)")
        
        self.tab_coord = QWidget()
        self.setup_coord_tab()
        self.tabs.addTab(self.tab_coord, "坐标控制 (IK)")
        
        # 3. 日志
        grp_log = QGroupBox("3. 日志")
        log_layout = QVBoxLayout()
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        log_layout.addWidget(self.txt_log)
        
        # 常用按钮
        btn_layout = QHBoxLayout()
        btn_reset = QPushButton("复位 Home")
        btn_reset.clicked.connect(self.reset_home)
        btn_suck = QPushButton("吸气")
        btn_suck.clicked.connect(lambda: self.send_udp_cmd(5, 1))
        btn_rel = QPushButton("释放")
        btn_rel.clicked.connect(lambda: self.send_udp_cmd(5, 3))
        btn_layout.addWidget(btn_reset)
        btn_layout.addWidget(btn_suck)
        btn_layout.addWidget(btn_rel)
        log_layout.addLayout(btn_layout)
        
        grp_log.setLayout(log_layout)
        right_layout.addWidget(grp_log)

    def setup_joint_tab(self):
        layout = QGridLayout()
        self.tab_joint.setLayout(layout)
        
        layout.addWidget(QLabel("J1 底座:"), 0, 0)
        self.sl_j1 = QSlider(Qt.Horizontal); self.sl_j1.setRange(-90, 90)
        self.sb_j1 = QDoubleSpinBox(); self.sb_j1.setRange(-90, 90)
        layout.addWidget(self.sl_j1, 0, 1); layout.addWidget(self.sb_j1, 0, 2)
        
        layout.addWidget(QLabel("J2 大臂:"), 1, 0)
        self.sl_j2 = QSlider(Qt.Horizontal); self.sl_j2.setRange(0, 180)
        self.sb_j2 = QDoubleSpinBox(); self.sb_j2.setRange(0, 180)
        layout.addWidget(self.sl_j2, 1, 1); layout.addWidget(self.sb_j2, 1, 2)
        
        layout.addWidget(QLabel("J3 小臂:"), 2, 0)
        self.sl_j3 = QSlider(Qt.Horizontal); self.sl_j3.setRange(-180, 180)
        self.sb_j3 = QDoubleSpinBox(); self.sb_j3.setRange(-180, 180)
        layout.addWidget(self.sl_j3, 2, 1); layout.addWidget(self.sb_j3, 2, 2)
        
        self.lbl_fk_xyz = QLabel("FK坐标: 待计算")
        self.lbl_fk_xyz.setStyleSheet("color: blue; font-weight: bold;")
        layout.addWidget(self.lbl_fk_xyz, 3, 0, 1, 3)

        self.btn_fk_exec = QPushButton("执行角度运动 (Execute)")
        self.btn_fk_exec.setStyleSheet("background-color: #ccffcc; font-weight: bold; height: 35px;")
        self.btn_fk_exec.clicked.connect(self.exec_fk_move)
        layout.addWidget(self.btn_fk_exec, 4, 0, 1, 3)

        # 绑定
        for w in [self.sl_j1, self.sl_j2, self.sl_j3]:
            w.valueChanged.connect(lambda v, w=w: self.on_joint_change('slider'))
        for w in [self.sb_j1, self.sb_j2, self.sb_j3]:
            w.valueChanged.connect(lambda v, w=w: self.on_joint_change('spin'))

    def setup_coord_tab(self):
        layout = QGridLayout()
        self.tab_coord.setLayout(layout)
        
        self.spin_x = QDoubleSpinBox(); self.spin_x.setRange(-60, 60); self.spin_x.setSingleStep(0.5)
        self.spin_y = QDoubleSpinBox(); self.spin_y.setRange(-60, 60); self.spin_y.setSingleStep(0.5)
        self.spin_z = QDoubleSpinBox(); self.spin_z.setRange(-60, 60); self.spin_z.setSingleStep(0.5)
        self.spin_x.setValue(HOME_X); self.spin_y.setValue(HOME_Y); self.spin_z.setValue(HOME_Z)
            
        layout.addWidget(QLabel("目标 X:"), 0, 0); layout.addWidget(self.spin_x, 0, 1)
        layout.addWidget(QLabel("目标 Y:"), 1, 0); layout.addWidget(self.spin_y, 1, 1)
        layout.addWidget(QLabel("目标 Z:"), 2, 0); layout.addWidget(self.spin_z, 2, 1)
        
        self.lbl_ik_result = QLabel("逆解: 待计算")
        layout.addWidget(self.lbl_ik_result, 3, 0, 1, 2)
        
        btn_calc = QPushButton("仅预览 (Preview)")
        btn_calc.clicked.connect(self.calc_ik_preview)
        layout.addWidget(btn_calc, 4, 0, 1, 2)
        
        self.btn_ik_exec = QPushButton("执行移动 (Execute)")
        self.btn_ik_exec.setStyleSheet("background-color: #ccffcc; font-weight: bold; height: 40px;")
        self.btn_ik_exec.clicked.connect(self.exec_ik_move)
        layout.addWidget(self.btn_ik_exec, 5, 0, 1, 2)

    # ================= 核心逻辑 =================
    
    def log(self, msg):
        t = time.strftime("%H:%M:%S")
        self.txt_log.append(f"[{t}] {msg}")
        print(msg)

    def update_speed_var(self, val):
        self.current_speed_us = val

    def toggle_lock(self):
        if self.btn_lock.isChecked():
            try:
                self.send_udp_cmd(0, 0) # Enable
                self.send_udp_cmd(4, START_DELAY_US) 
                self.lock_status = True
                self.btn_lock.setText("已锁定 (解锁)")
                self.btn_lock.setStyleSheet("background-color: #ccffcc;")
                self.last_steps = [0, 0, 0] 
                self.log("电机已锁定。")
                self.on_joint_change('spin') # Sync UI
            except Exception as e:
                self.log(f"连接错误: {e}")
                self.btn_lock.setChecked(False)
        else:
            self.send_udp_cmd(0, 1) # Disable
            self.lock_status = False
            self.btn_lock.setText("锁定控制")
            self.btn_lock.setStyleSheet("background-color: #ffcccc;")
            self.log("电机已释放。")

    def reset_home(self):
        self.log("复位中...")
        self.sb_j1.setValue(HOME_THETA_1)
        self.sb_j2.setValue(HOME_THETA_2)
        self.sb_j3.setValue(HOME_THETA_3)
        self.calc_ik_preview()
        if self.lock_status:
            self.start_smooth_move(HOME_THETA_1, HOME_THETA_2, HOME_THETA_3)

    # --- 关节控制 ---
    def on_joint_change(self, src):
        if src == 'slider':
            self.sb_j1.setValue(self.sl_j1.value())
            self.sb_j2.setValue(self.sl_j2.value())
            self.sb_j3.setValue(self.sl_j3.value())
        else:
            self.sl_j1.setValue(int(self.sb_j1.value()))
            self.sl_j2.setValue(int(self.sb_j2.value()))
            self.sl_j3.setValue(int(self.sb_j3.value()))

        t1, t2, t3 = self.sb_j1.value(), self.sb_j2.value(), self.sb_j3.value()
        self.preview_angles = [t1, t2, t3]
        
        x, y, z = self.solve_fk(t1, t2, t3)
        self.lbl_fk_xyz.setText(f"FK预览: {x:.1f}, {y:.1f}, {z:.1f}")
        self.update_plot()

    def exec_fk_move(self):
        if not self.lock_status:
            QMessageBox.warning(self, "未锁定", "请先锁定电机！")
            return
        t1, t2, t3 = self.preview_angles
        self.start_smooth_move(t1, t2, t3)

    # --- 坐标控制 ---
    def calc_ik_preview(self):
        res = self.solve_ik(self.spin_x.value(), self.spin_y.value(), self.spin_z.value())
        if res:
            t1, t2, t3 = res
            self.lbl_ik_result.setText(f"逆解: {t1:.1f}, {t2:.1f}, {t3:.1f}")
            self.preview_angles = [t1, t2, t3]
            self.update_plot()
        else:
            self.lbl_ik_result.setText("不可达")

    def exec_ik_move(self):
        res = self.solve_ik(self.spin_x.value(), self.spin_y.value(), self.spin_z.value())
        if not res: return
        t1, t2, t3 = res
        
        # 更新关节UI
        self.sb_j1.setValue(t1); self.sl_j1.setValue(int(t1))
        self.sb_j2.setValue(t2); self.sl_j2.setValue(int(t2))
        self.sb_j3.setValue(t3); self.sl_j3.setValue(int(t3))
        
        if self.lock_status:
            self.start_smooth_move(t1, t2, t3)

    # ================= 启动线程 =================
    def start_smooth_move(self, t1, t2, t3):
        self.set_buttons_enabled(False)
        self.worker = MovementWorker(self, (t1, t2, t3))
        self.worker.log_signal.connect(self.log)
        self.worker.finished_signal.connect(lambda: self.set_buttons_enabled(True))
        self.worker.start()

    def set_buttons_enabled(self, val):
        self.btn_fk_exec.setEnabled(val)
        self.btn_ik_exec.setEnabled(val)
        self.btn_fk_exec.setText("执行中..." if not val else "执行角度运动 (Execute)")

    # ================= 算法 & 通信 =================
    def solve_fk(self, t1, t2, t3):
        rad1, rad2, rad3 = map(math.radians, [t1, t2, t3])
        r_wrist = L2_ARM * math.cos(rad2) + L3_FOREARM * math.cos(rad3)
        z_wrist = L1_BASE_H + L2_ARM * math.sin(rad2) + L3_FOREARM * math.sin(rad3)
        r_end = r_wrist + L4_HAND_H
        z_end = z_wrist - L4_HAND_V
        return r_end * math.cos(rad1), r_end * math.sin(rad1), z_end

    def solve_ik(self, x, y, z):
        t1 = math.degrees(math.atan2(y, x))
        r_end, z_end = math.sqrt(x**2 + y**2), z
        r_wrist, z_wrist = r_end - L4_HAND_H, z_end + L4_HAND_V
        dz, dr = z_wrist - L1_BASE_H, r_wrist
        hyp = math.sqrt(dr**2 + dz**2)
        
        if hyp > (L2_ARM + L3_FOREARM) or hyp < abs(L2_ARM - L3_FOREARM): return None
        try:
            cos_alpha = (L2_ARM**2 + hyp**2 - L3_FOREARM**2) / (2 * L2_ARM * hyp)
            alpha = math.acos(cos_alpha) if abs(cos_alpha) <= 1 else 0
            beta = math.atan2(dz, dr)
            t2 = math.degrees(beta + alpha)
            
            r_elbow = L2_ARM * math.cos(math.radians(t2))
            z_elbow = L2_ARM * math.sin(math.radians(t2))
            t3 = math.degrees(math.atan2(dz - z_elbow, dr - r_elbow))
            return (t1, t2, t3)
        except: return None

    def send_udp_cmd(self, cmd, val):
        try:
            msg = f"{cmd},{val}"
            self.udp_socket.sendto(msg.encode(), (self.ip_input.text(), int(self.port_input.text())))
        except: pass

    def update_plot(self):
        self.canvas.axes.clear()
        t1, t2, t3 = self.preview_angles
        rad1, rad2, rad3 = map(math.radians, [t1, t2, t3])
        
        x1, y1, z1 = 0, 0, L1_BASE_H
        r2 = L2_ARM * math.cos(rad2)
        x2, y2, z2 = x1 + r2 * math.cos(rad1), y1 + r2 * math.sin(rad1), z1 + L2_ARM * math.sin(rad2)
        
        dr3, dz3 = L3_FOREARM * math.cos(rad3), L3_FOREARM * math.sin(rad3)
        x3, y3, z3 = x2 + dr3 * math.cos(rad1), y2 + dr3 * math.sin(rad1), z2 + dz3
        
        dr4, dz4 = L4_HAND_H, -L4_HAND_V
        x4, y4, z4 = x3 + dr4 * math.cos(rad1), y3 + dr4 * math.sin(rad1), z3 + dz4
        
        points = [(0,0,0), (x1,y1,z1), (x2,y2,z2), (x3,y3,z3), (x4,y4,z4)]
        xs, ys, zs = zip(*points)
        self.canvas.axes.plot(xs, ys, zs, 'o-', linewidth=4, markersize=8, label='Arm')
        self.canvas.axes.plot([x3, x4], [y3, y4], [z3, z4], 'r-', linewidth=5, label='Hand')
        self.canvas.axes.set_xlim(-40, 40); self.canvas.axes.set_ylim(-40, 40); self.canvas.axes.set_zlim(0, 50)
        self.canvas.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = RobotController()
    window.show()
    sys.exit(app.exec_())