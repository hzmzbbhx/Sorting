import random
import json
import socket
import sys
import time  # 新增: 用于延时
import os

import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt, QThread, pyqtSignal  # 新增: 线程信号
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QSlider

import p_fkdh as fk
from mplwidget import MplWidget

# ==========================================
# 【新增 1】: 导入视觉窗口类
# 注意：monitor.py 必须在同一目录下
# ==========================================

from monitor import MainWindow as VisionWindow



# ==========================================
# 【新增 2】: 分拣动作工作线程
# 负责执行一连串的机械臂动作，防止主界面卡死
# ==========================================
class SortingWorker(QThread):
    finished_signal = pyqtSignal()  # 动作完成信号

    def __init__(self, target_ip, target_port):
        super().__init__()
        self.ip = target_ip
        self.port = target_port
        self.is_running = True
        # 在线程内部创建独立的UDP套接字，避免冲突
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send_cmd(self, cmd_str):
        """发送指令辅助函数"""
        if self.is_running:
            print(f"[Worker] 发送指令: {cmd_str}")
            self.sock.sendto(cmd_str.encode('utf-8'), (self.ip, self.port))
            time.sleep(0.1)  # 指令发送间隔

    def run(self):
        print("--- 开始执行自动化分拣 ---")

        # ============================================
        # 【自定义动作序列】
        # 请根据您的机械臂实际情况修改这里的坐标(步数)
        # 格式: "7, 轴1步数, 轴2步数, 轴3步数"
        # ============================================

        # 1. 移动到异常物体上方
        # 假设物体坐标对应的步数是: 1轴200, 2轴100, 3轴0
        self.send_cmd("7,200,100,0")
        time.sleep(2.5)  # 等待运动完成

        # 2. 下降 (Z轴向下，假设轴2和轴3配合下降)
        self.send_cmd("7,0,50,50")
        time.sleep(1.5)

        # 3. 开启吸盘 (cmd=5, 1=吸气)
        self.send_cmd("5,1")
        time.sleep(1.0)

        # 4. 抬起 (反向运动)
        self.send_cmd("7,0,-50,-50")
        time.sleep(1.5)

        # 5. 搬运到放置区 (假设在另一侧)
        self.send_cmd("7,-400,0,0")  # 1轴反转
        time.sleep(3.0)

        # 6. 放下物体 (cmd=5, 3=排气释放)
        self.send_cmd("5,3")
        time.sleep(0.5)

        # 7. 机械臂复位 (回到初始状态)
        # 此时累计步数: 1轴(200-400)=-200, 2轴(100+50-50)=100, 3轴(0+50-50)=0
        # 需要回正: 1轴+200, 2轴-100, 3轴0
        self.send_cmd("7,200,-100,0")
        time.sleep(2.5)

        print("--- 分拣完成 ---")
        self.finished_signal.emit()

    def stop(self):
        self.is_running = False


class RobotArm(object):
    j1, j2, j3, j4, x4, y4, z4, Tm = 0, 0, 0, 0, 0, 0, 0, 0
    xt, yt, zt = 0, 0, 0
    re_set = 0
    objek = {}

    def __init__(self, window):
        # 初始化变量
        self.vision_window = None  # 保存视觉窗口实例
        self.is_sorting = False    # 分拣状态标志位
        
        # 设置界面
        self.__set_ui(window)
        # 创建网络UDP功能
        self.__create_socket()
        # 初始化
        self.__init_state()
        # 处理
        self.fkmotion()
        self.pushButton_reset.clicked.connect(self.__init_state)
        self.pushButton_runik.clicked.connect(self.start_ik)
        self.pushButton_rand.clicked.connect(self.randomtarget)
        
        # 【新增 3】添加启动视觉界面的按钮
        self.__add_vision_ui(window)

    def __add_vision_ui(self, MainWindow):
        """在界面右下角添加视觉启动按钮"""
        self.btn_open_vision = QtWidgets.QPushButton(self.centralwidget)
        # 位置放在“锁定”按钮下方
        self.btn_open_vision.setGeometry(QtCore.QRect(890, 520, 131, 40))
        self.btn_open_vision.setText("启动视觉检测")
        # 设置为蓝色背景
        self.btn_open_vision.setStyleSheet("background-color: #0078d7; color: white; font-weight: bold; font-size: 14px;")
        self.btn_open_vision.clicked.connect(self.open_vision_system)

    def open_vision_system(self):
        """启动 monitor.py 的窗口"""
        if VisionWindow is None:
            QtWidgets.QMessageBox.warning(self.centralwidget, "错误", "未找到 monitor.py，无法启动视觉系统。")
            return

        if self.vision_window is None:
            self.vision_window = VisionWindow()
            # 【关键】连接信号：当视觉线程发出结果时，调用 self.handle_vision_result
            self.vision_window.thread.result_signal.connect(self.handle_vision_result)
        
        # 调整窗口位置，避免完全遮挡
        self.vision_window.move(100, 100)
        self.vision_window.show()
        print("视觉系统已启动，等待 NG 信号...")

    def handle_vision_result(self, status, score, heatmap):
        """
        处理 monitor.py 发过来的信号
        """
        # 1. 只有当状态是 NG，且机械臂当前没在忙，才执行
        if status == "NG" and not self.is_sorting:
            # 2. 检查必须处于“锁定”状态（说明已连接ESP32）
            if not self.lock_btn.current_status:
                print(f"[忽略] 视觉发现异常(Score:{score:.2f})，但机械臂未锁定。")
                return

            print(f"!!! 视觉报警: {status} (分数: {score:.2f}) -> 触发分拣")
            self.start_sorting_sequence()

    def start_sorting_sequence(self):
        """启动分拣线程"""
        self.is_sorting = True  # 标记为忙碌
        
        # 获取当前填写的 IP 和 端口
        dest_ip = self.ip_edit.text()
        try:
            dest_port = int(self.port_edit.text())
        except:
            dest_port = 7788

        # 创建并启动工作线程
        self.worker = SortingWorker(dest_ip, dest_port)
        self.worker.finished_signal.connect(self.on_sorting_finished)
        self.worker.start()

    def on_sorting_finished(self):
        """分拣结束后的回调"""
        self.is_sorting = False  # 解除忙碌状态
        print("机械臂已复位，系统待机中...")

    # ==========================================
    # 以下为原有代码，保持不变
    # ==========================================

    def __create_socket(self):
        """创建UDP套接字"""
        # 1. 创建udp套接字
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # 2. 发送数据到指定的电脑上的指定程序中
        # self.udp_socket.sendto("hello".encode('utf-8'), ("192.168.1.2", 8800))

    def __set_ui(self, MainWindow):
        MainWindow.setWindowTitle("机械臂D1-桌面控制器 (集成视觉版)")
        MainWindow.setWindowIcon(QIcon("./logo.ico"))
        MainWindow.resize(1075, 576)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        # ---------------------- 左侧 ----------------------
        # 创建matplotlib生成的canvas嵌入到QWidget中
        self.mpl_widget = MplWidget(self.centralwidget)
        self.mpl_widget.setEnabled(True)
        self.mpl_widget.setGeometry(QtCore.QRect(20, 30, 581, 491))
        self.mpl_widget.setMouseTracking(True)
        self.mpl_widget.setTabletTracking(True)
        self.mpl_widget.setObjectName("MplWidget")

        # ---------------------- 正向运动学 ----------------------
        label_fk = QtWidgets.QLabel(self.centralwidget)
        label_fk.setGeometry(QtCore.QRect(620, 40, 181, 16))
        font = QtGui.QFont()
        font.setPointSize(12)
        label_fk.setFont(font)
        label_fk.setText("正向运动学")

        # 文字
        labels1 = QtWidgets.QLabel(self.centralwidget)
        labels1.setGeometry(QtCore.QRect(620, 90, 91, 16))
        labels1.setText("1轴角度")
        labels2 = QtWidgets.QLabel(self.centralwidget)
        labels2.setGeometry(QtCore.QRect(620, 120, 91, 16))
        labels2.setText("2轴角度")
        labels3 = QtWidgets.QLabel(self.centralwidget)
        labels3.setGeometry(QtCore.QRect(620, 150, 91, 16))
        labels3.setText("3轴角度")
        labels4 = QtWidgets.QLabel(self.centralwidget)
        labels4.setGeometry(QtCore.QRect(620, 180, 91, 16))
        labels4.setText("4轴角度")

        label_iks_3 = QtWidgets.QLabel(self.centralwidget)
        label_iks_3.setGeometry(QtCore.QRect(706, 69, 61, 16))
        label_iks_3.setText("真实角度")
        label_iks_4 = QtWidgets.QLabel(self.centralwidget)
        label_iks_4.setGeometry(QtCore.QRect(770, 70, 71, 16))
        label_iks_4.setText("目标角度")

        # 真实角度
        self.label_j1 = QtWidgets.QLabel(self.centralwidget)
        self.label_j1.setGeometry(QtCore.QRect(702, 90, 61, 21))
        self.label_j1.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_j2 = QtWidgets.QLabel(self.centralwidget)
        self.label_j2.setGeometry(QtCore.QRect(702, 120, 61, 21))
        self.label_j2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_j3 = QtWidgets.QLabel(self.centralwidget)
        self.label_j3.setGeometry(QtCore.QRect(702, 150, 61, 21))
        self.label_j3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_j4 = QtWidgets.QLabel(self.centralwidget)
        self.label_j4.setGeometry(QtCore.QRect(702, 180, 61, 21))
        self.label_j4.setFrameShape(QtWidgets.QFrame.StyledPanel)

        # 目标角度
        self.doubleSpinBox1 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox1.setGeometry(QtCore.QRect(772, 90, 62, 22))
        self.doubleSpinBox1.setAccelerated(True)
        self.doubleSpinBox1.setMinimum(0.0)
        self.doubleSpinBox1.setMaximum(180.0)
        self.doubleSpinBox2 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox2.setGeometry(QtCore.QRect(772, 120, 62, 22))
        self.doubleSpinBox2.setAccelerated(True)
        self.doubleSpinBox2.setMinimum(-90.0)
        self.doubleSpinBox2.setMaximum(90.0)
        self.doubleSpinBox3 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox3.setGeometry(QtCore.QRect(772, 150, 62, 22))
        self.doubleSpinBox3.setAccelerated(True)
        self.doubleSpinBox3.setMinimum(-90.0)
        self.doubleSpinBox3.setMaximum(90.0)

        # 重置按钮
        self.pushButton_reset = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_reset.setGeometry(QtCore.QRect(860, 90, 161, 111))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.pushButton_reset.setFont(font)
        self.pushButton_reset.setText("重置")

        # ---------------------- 逆向运动学 ----------------------
        label_ik = QtWidgets.QLabel(self.centralwidget)
        label_ik.setGeometry(QtCore.QRect(620, 240, 191, 16))
        font = QtGui.QFont()
        font.setPointSize(12)
        label_ik.setFont(font)
        label_ik.setText("逆运动学")

        labels1_ik = QtWidgets.QLabel(self.centralwidget)
        labels1_ik.setGeometry(QtCore.QRect(620, 290, 91, 16))
        labels1_ik.setText("位置X")
        labels2_ik = QtWidgets.QLabel(self.centralwidget)
        labels2_ik.setGeometry(QtCore.QRect(620, 320, 91, 16))
        labels2_ik.setText("位置Y")
        labels3_ik = QtWidgets.QLabel(self.centralwidget)
        labels3_ik.setGeometry(QtCore.QRect(620, 350, 91, 16))
        labels3_ik.setText("位置Z")

        label_iks = QtWidgets.QLabel(self.centralwidget)
        label_iks.setGeometry(QtCore.QRect(701, 269, 61, 16))
        label_iks.setText("真实位置")
        label_iks_2 = QtWidgets.QLabel(self.centralwidget)
        label_iks_2.setGeometry(QtCore.QRect(777, 268, 71, 16))
        label_iks_2.setText("目标位置")

        # 真实位置
        self.label_x4 = QtWidgets.QLabel(self.centralwidget)
        self.label_x4.setGeometry(QtCore.QRect(697, 290, 61, 21))
        self.label_x4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_y4 = QtWidgets.QLabel(self.centralwidget)
        self.label_y4.setGeometry(QtCore.QRect(697, 320, 61, 21))
        self.label_y4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_z4 = QtWidgets.QLabel(self.centralwidget)
        self.label_z4.setGeometry(QtCore.QRect(697, 350, 61, 21))
        self.label_z4.setFrameShape(QtWidgets.QFrame.StyledPanel)

        # 目标位置
        self.doubleSpinBox_ik1 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_ik1.setGeometry(QtCore.QRect(777, 290, 62, 22))
        self.doubleSpinBox_ik1.setAccelerated(True)
        self.doubleSpinBox_ik1.setMinimum(-99.0)
        self.doubleSpinBox_ik2 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_ik2.setGeometry(QtCore.QRect(777, 320, 62, 22))
        self.doubleSpinBox_ik2.setAccelerated(True)
        self.doubleSpinBox_ik3 = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_ik3.setGeometry(QtCore.QRect(777, 350, 62, 22))
        self.doubleSpinBox_ik3.setAccelerated(True)

        # 运行按钮
        self.pushButton_runik = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_runik.setGeometry(QtCore.QRect(860, 310, 161, 61))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.pushButton_runik.setFont(font)
        self.pushButton_runik.setText("运行")

        # 随机位置按钮
        self.pushButton_rand = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_rand.setGeometry(QtCore.QRect(860, 272, 161, 31))
        self.pushButton_rand.setText("随机位置")

        # ---------------------- 步进电机转动 ----------------------
        stepper_motor_label = QtWidgets.QLabel(self.centralwidget)
        stepper_motor_label.setGeometry(QtCore.QRect(620, 410, 191, 16))
        font = QtGui.QFont()
        font.setPointSize(12)
        stepper_motor_label.setFont(font)
        stepper_motor_label.setText("步进电机转动")

        # 目标IP
        ip_label = QtWidgets.QLabel(self.centralwidget)
        ip_label.setGeometry(QtCore.QRect(740, 413, 91, 16))
        ip_label.setText("IP:")
        self.ip_edit = QtWidgets.QLineEdit(self.centralwidget)
        self.ip_edit.setText("192.168.1.12") # 默认填入调试好的IP
        self.ip_edit.setGeometry(QtCore.QRect(760, 413, 91, 16))
        # 目标PORT
        ip_port = QtWidgets.QLabel(self.centralwidget)
        ip_port.setGeometry(QtCore.QRect(860, 413, 91, 16))
        ip_port.setText("PORT:")
        self.port_edit = QtWidgets.QLineEdit(self.centralwidget)
        self.port_edit.setText("7788")
        self.port_edit.setGeometry(QtCore.QRect(890, 413, 40, 16))
        # 锁定位置按钮
        self.lock_btn = QtWidgets.QPushButton(self.centralwidget)
        self.lock_btn.setText("锁定")
        self.lock_btn.setGeometry(QtCore.QRect(950, 410, 70, 25))
        self.lock_btn.current_status = False
        self.lock_btn.clicked.connect(self.lock_btn_callback)

        stepper_motor_axis_1 = QtWidgets.QLabel(self.centralwidget)
        stepper_motor_axis_1.setGeometry(QtCore.QRect(620, 440, 91, 16))
        stepper_motor_axis_1.setText("1轴电机")
        self.sl_1 = QSlider(Qt.Horizontal)
        self.sl_1.setParent(self.centralwidget)
        self.sl_1.setGeometry(QtCore.QRect(690, 440, 320, 16))
        self.sl_1.setMinimum(-50)
        self.sl_1.setMaximum(50)
        self.sl_1.setValue(0)
        self.sl_1_label = QtWidgets.QLabel(self.centralwidget)
        self.sl_1_label.setText("0")
        self.sl_1_label.setGeometry(QtCore.QRect(1020, 438, 20, 16))
        self.sl_1.valueChanged.connect(lambda x: self.stepper_motor_move(1, x, self.sl_1_label))

        stepper_motor_axis_2 = QtWidgets.QLabel(self.centralwidget)
        stepper_motor_axis_2.setGeometry(QtCore.QRect(620, 470, 91, 16))
        stepper_motor_axis_2.setText("2轴电机")
        self.sl_2 = QSlider(Qt.Horizontal)
        self.sl_2.setParent(self.centralwidget)
        self.sl_2.setGeometry(QtCore.QRect(690, 470, 320, 16))
        self.sl_2.setMinimum(-50)
        self.sl_2.setMaximum(50)
        self.sl_2.setValue(0)
        self.sl_2_label = QtWidgets.QLabel(self.centralwidget)
        self.sl_2_label.setText("0")
        self.sl_2_label.setGeometry(QtCore.QRect(1020, 470, 20, 16))
        self.sl_2.valueChanged.connect(lambda x: self.stepper_motor_move(2, x, self.sl_2_label))

        stepper_motor_axis_3 = QtWidgets.QLabel(self.centralwidget)
        stepper_motor_axis_3.setGeometry(QtCore.QRect(620, 500, 91, 16))
        stepper_motor_axis_3.setText("3轴电机")
        self.sl_3 = QSlider(Qt.Horizontal)
        self.sl_3.setParent(self.centralwidget)
        self.sl_3.setGeometry(QtCore.QRect(690, 500, 320, 16))
        self.sl_3.setMinimum(-50)
        self.sl_3.setMaximum(50)
        self.sl_3.setValue(0)
        self.sl_3_label = QtWidgets.QLabel(self.centralwidget)
        self.sl_3_label.setText("0")
        self.sl_3_label.setGeometry(QtCore.QRect(1020, 500, 20, 16))
        self.sl_3.valueChanged.connect(lambda x: self.stepper_motor_move(3, x, self.sl_3_label))

        # 其它
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1075, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def stepper_motor_move(self, *args):
        """滑动电机滑块时的回调函数"""
        # print(args)
        axis, number, label_obj = args
        label_obj.setText(str(number))
        if self.lock_btn.current_status:
            dest_ip = self.ip_edit.text()
            dest_port = int(self.port_edit.text())
            cmd = "%d,%d" % (axis, number)
            self.udp_socket.sendto(cmd.encode('utf-8'), (dest_ip, dest_port))

    def lock_btn_callback(self, *args):
        """ip、port、机械臂位置锁定"""
        print(args)
        dest_ip = self.ip_edit.text()
        dest_port = int(self.port_edit.text())
        if self.lock_btn.current_status is False:
            # 滑块恢复0
            self.sl_1.setValue(0)
            self.sl_2.setValue(0)
            self.sl_3.setValue(0)
            # 锁定ip、port
            self.ip_edit.setEnabled(False)
            self.port_edit.setEnabled(False)
            # 更改状态
            self.lock_btn.current_status = True
            self.lock_btn.setText("解除锁定")
            # 发送锁定命令
            cmd = "0,0"
            self.udp_socket.sendto(cmd.encode('utf-8'), (dest_ip, dest_port))
        else:
            self.ip_edit.setEnabled(True)
            self.port_edit.setEnabled(True)
            self.lock_btn.current_status = False
            self.lock_btn.setText("锁定")
            # 发送解锁命令
            cmd = "0,1"
            self.udp_socket.sendto(cmd.encode('utf-8'), (dest_ip, dest_port))

    def konek2robot(self):
        desc = "Detected.. "
        for x in range(len(self.objek)):
            desc = desc + "\n" + str(x) + ": objek: " + str(self.objek[x]['nama']) + " Pos: " + str(
                self.objek[x]['centerbbox'])
        # self.label_RStatus.setText(str(len(self.objek)))
        self.label_RStatus.setText(desc)
        print(self.objek)

    def __init_state(self):
        self.re_set = 1
        self.j1 = 90
        self.j2 = 60
        self.j3 = -90
        self.j4 = -self.j2 - self.j3
        self.doubleSpinBox1.setProperty("value", self.j1)
        self.doubleSpinBox2.setProperty("value", self.j2)
        self.doubleSpinBox3.setProperty("value", self.j3)
        self.label_j1.setText(str(round(self.j1, 2)))
        self.label_j2.setText(str(round(self.j2, 2)))
        self.label_j3.setText(str(round(self.j3, 2)))
        self.label_j4.setText(str(round(self.j4, 2)))
        self.mpl_widget.canvas.axes.view_init(20, 320)
        self.drawfk(self.j1, self.j2, self.j3)

    def randomtarget(self):
        self.doubleSpinBox_ik1.setValue(random.randint(-40, 40))
        self.doubleSpinBox_ik2.setValue(random.randint(5, 40))
        self.doubleSpinBox_ik3.setValue(random.randint(5, 40))

    def fkmotion(self):
        self.doubleSpinBox1.valueChanged.connect(self.eepos_fk)
        self.doubleSpinBox2.valueChanged.connect(self.eepos_fk)
        self.doubleSpinBox3.valueChanged.connect(self.eepos_fk)

    def eepos_fk(self):
        self.j1 = self.doubleSpinBox1.value()
        self.j2 = self.doubleSpinBox2.value()
        self.j3 = self.doubleSpinBox3.value()
        self.j4 = -self.j2 - self.j3
        self.label_j1.setText(str(round(self.j1, 2)))
        self.label_j2.setText(str(round(self.j2, 2)))
        self.label_j3.setText(str(round(self.j3, 2)))
        self.label_j4.setText(str(round(self.j4, 2)))

        self.drawfk(self.j1, self.j2, self.j3)

    def start_ik(self):
        self.xt = self.doubleSpinBox_ik1.value()
        self.yt = self.doubleSpinBox_ik2.value()
        self.zt = self.doubleSpinBox_ik3.value()
        # self.xt=-53.77
        # self.yt=31.04
        # self.zt=15.17
        self.re_set = 0
        self.pinj_ik()

    def pinj_ik(self):
        if self.re_set == 0:
            x_start = self.x4
            y_start = self.y4
            z_start = self.z4
            ptarget = np.vstack([self.xt, self.yt, self.zt])
            pstart = np.vstack([x_start, y_start, z_start])
            delta = fk.divelo(ptarget, pstart)
            step = 5
            EucXYZ = delta[4]
            if abs(EucXYZ) > 5:
                pembagi_step = abs(EucXYZ) / step
                dXYZ = delta[0:3] / pembagi_step
            else:
                dXYZ = delta[0:3]

            if abs(EucXYZ) <= 0.01:
                self.mpl_widget.msgwarning()
            else:
                Jac = fk.jacobian(self.Tm)
                Jac_Inv = fk.PinvJac(Jac)
                dTheta = Jac_Inv @ dXYZ
                dTheta1 = np.rad2deg(dTheta[0])
                dTheta2 = np.rad2deg(dTheta[1])
                dTheta3 = np.rad2deg(dTheta[2])
                self.j1 = self.j1 + dTheta1
                self.j2 = self.j2 + dTheta2
                self.j3 = self.j3 + dTheta3
                self.j4 = -self.j2 - self.j3
                self.label_j1.setText(str(round(self.j1[0], 2)))
                self.label_j2.setText(str(round(self.j2[0], 2)))
                self.label_j3.setText(str(round(self.j3[0], 2)))
                self.label_j4.setText(str(round(self.j4[0], 2)))
                self.drawfk(self.j1, self.j2, self.j3)
                QApplication.processEvents()
                self.pinj_ik()

    def drawfk(self, a, b, c):
        """绘制机械臂新的位置"""

        j = fk.dh_par(a, b, c)
        self.Tm = fk.dh_kine(j)
        ee = fk.el_xyzpos(self.Tm)
        p0, p1, p2, p3, p4, p5 = fk.el_pos2base(self.Tm)
        X1, Y1, Z1 = ee[0, 0:3], ee[1, 0:3], ee[2, 0:3]
        X2, Y2, Z2 = ee[0, 2:4], ee[1, 2:4], ee[2, 2:4]
        X3, Y3, Z3 = ee[0, 3:5], ee[1, 3:5], ee[2, 3:5]
        X4, Y4, Z4 = ee[0, 4:6], ee[1, 4:6], ee[2, 4:6]

        self.x4 = X4[1]
        self.y4 = Y4[1]
        self.z4 = Z4[1]

        # self.doubleSpinBox_ik1.setValue(round(self.x4,2))
        # self.doubleSpinBox_ik2.setValue(round(self.y4,2))
        # self.doubleSpinBox_ik3.setValue(round(self.z4,2))

        self.label_x4.setText(str(round(self.x4, 2)))
        self.label_y4.setText(str(round(self.y4, 2)))
        self.label_z4.setText(str(round(self.z4, 2)))

        self.mpl_widget.canvas.axes.clear()
        self.mpl_widget.canvas.axes.plot(X1, Y1, Z1, color='green', marker='o', linestyle='solid', linewidth=5, markersize=10)
        self.mpl_widget.canvas.axes.plot(X2, Y2, Z2, color='red', marker='o', linestyle='solid', linewidth=5, markersize=10)
        self.mpl_widget.canvas.axes.plot(X3, Y3, Z3, color='blue', marker='o', linestyle='solid', linewidth=5, markersize=10)
        self.mpl_widget.canvas.axes.plot(X4, Y4, Z4, color='purple', marker="h", linestyle='solid', linewidth=5, markersize=10)

        self.mpl_widget.canvas.axes.text(self.x4, self.y4, self.z4, '({:.2f}, {:.2f}, {:.2f})'.format(self.x4, self.y4, self.z4), weight='bold', fontsize=12, )
        self.mpl_widget.defcanvas()
        self.mpl_widget.canvas.draw()

    def read_json(self):
        # read file
        with open('dataobjek.json', 'r') as myfile:
            data = myfile.read()
        # parse file
        obj = json.loads(data)
        self.objek = obj['objek']
        self.label_RStatus.setText("json loaded")

    def write_json(self, data, filename='dataobjek.json'):
        with open(filename, 'w') as f:
            json.dump(data, f)

    def emptyjson(self, filename='dataobjek.json'):
        dataobj = {}
        json_object = json.dumps(dataobj)
        write_json(json_object, filename)


class kinematic():
    def __init__(self):
        self.a = 12
        self.b = 3


if __name__ == '__main__':
    # 创建app对象
    app = QtWidgets.QApplication(sys.argv)

    # 创建主窗口
    main_window = QtWidgets.QMainWindow()
    ui = RobotArm(main_window)
    main_window.show()

    # app无限循环
    app.exec_()