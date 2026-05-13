# 机械臂智能分拣系统技术文档（YOLOv8定位 + AnomalyCLIP识别）

## 1. 系统概述（System Overview）

本系统是一套集**AI视觉识别、异常检测、机械臂自动分拣**于一体的智能质检解决方案，专为复杂工业场景下的微小瑕疵检测与分拣设计（如扑克牌出千标记识别、精密零件划痕筛选等）。系统采用 **Two-Stage（两阶段）级联架构**，通过“先定位后质检”的核心逻辑，实现低数据需求下的高精度、实时性分拣。

### 核心优势

- **少样本适配**：仅需少量正常样本（10-20张）和个位数异常样本，即可通过LoRA微调完成特定场景适配，无需海量标注数据。

- **高精度检测**：结合YOLOv8的目标定位能力与AnomalyCLIP的像素级异常识别能力，缺陷检测准确率可达工业级标准。

- **实时性强**：YOLOv8快速定位（毫秒级）+ AnomalyCLIP轻量化推理 + S-Curve平滑运动控制，满足生产线实时分拣需求。

- **全流程自动化**：从摄像头采集、目标裁剪、异常判定，到机械臂抓取、废料区投放，全程无需人工干预。

## 2. 核心技术原理（Core Technologies）

### 2.1 第一阶段：目标定位（YOLOv8）

#### 技术选型

采用 Ultralytics YOLOv8 轻量级目标检测模型，专注于“从复杂背景中找到目标”。

#### 核心作用

- **抗干扰过滤**：在包含传送带、桌面杂物的复杂画面中，精准框选待检测物体（如扑克牌、零件），剔除背景噪声（光照变化、环境杂物）。

- **多目标并行处理**：支持同时检测视野内多个物体，为后续并行质检与分拣提供基础。

- **输出结果**：目标边界框坐标 `(x1, y1, x2, y2)`，用于后续图像裁剪（ROI提取）。

#### 关键参数

- 预训练模型：[yolov8n.pt](yolov8n.pt)（兼顾速度与精度）

- 置信度阈值：`CONF_THRESH = 0.5`（过滤低精度检测结果）

- 输入图像尺寸：640×640（适配工业相机采集分辨率）

- 数据增强：开启 `augment=True`，提升模型泛化能力。

### 2.2 第二阶段：异常检测（AnomalyCLIP + LoRA）

作为系统核心“质检大脑”，基于 OpenAI CLIP（ViT-L/14@336px）进行工业场景定制化改造，通过 **LoRA低秩自适应微调** 实现少样本适配。

#### 2.2.1 AnomalyCLIP 核心改造

为解决原版CLIP“重语义、轻细节”的缺陷，针对性优化两点：

##### （1）DAPM 双路径注意力机制

- **设计逻辑**：在ViT视觉编码器深层（第20层后）引入 **Value-Value Attention**，强制模型关注特征间的纹理相关性，而非仅依赖空间语义。

- **数学原理**：通过计算  $V \cdot V^T$ （类似Gram矩阵），强化对划痕、异色、污渍等“破坏纹理连续性”的微小缺陷的敏感度。

- **作用**：让模型具备“显微镜级”细节识别能力，突破传统CLIP对同类物体缺陷区分能力弱的瓶颈。

##### （2）Deep Prompt Learning 深度提示学习

- **创新点**：不仅在输入端注入文本提示（如“normal object”“damaged object”），还在Transformer每一层插入**可学习的上下文向量（compound_prompts_deeper）**。

- **作用**：让模型在特征提取的全流程中持续强化“正常/异常”的概念认知，实现视觉特征与文本语义的深层对齐。

#### 2.2.2 LoRA 低秩自适应微调

针对全量微调大模型的痛点（显存需求高、数据依赖大、易遗忘预训练知识），采用参数高效微调策略：

- **核心逻辑**：冻结CLIP原始参数，仅在Attention层（qkv、proj、c_fc、c_proj）旁路插入低秩矩阵  $A$  和  $B$  进行训练。

- **技术优势**：

    - 少样本收敛：10-20张样本即可完成适配，无需海量标注数据。

    - 参数效率：训练参数量减少90%以上，降低硬件门槛。

    - 保留泛化性：不破坏CLIP的通用视觉特征，可识别未见过的新型缺陷。

#### 2.2.3 多损失函数组合

为兼顾“整图分类准确性”与“像素级缺陷定位精度”，设计三重损失函数：

|损失函数类型|作用|
|---|---|
|Image Cross Entropy Loss|确保整图分类（OK/NG）结果正确|
|Focal Loss|解决像素不平衡问题（缺陷像素占比＜1%），聚焦难分类像素|
|Binary Dice Loss|优化预测掩码与真实掩码（Ground Truth）的IoU重叠度|
## 3. 系统架构（System Architecture）

### 3.1 硬件层（Hardware Layer）

|组件|规格/型号|功能描述|
|---|---|---|
|机械臂|D1 三轴机械臂（SCARA/Serial Hybrid结构）|执行抓取、搬运、释放动作，含3个步进电机驱动|
|控制器|ESP32（MicroPython固件）|接收UDP指令，控制电机与末端执行器|
|末端执行器|气泵 + 电磁阀|实现吸气抓取、释放功能|
|视觉传感器|USB工业相机/网络摄像头|实时采集画面，分辨率1280×720|
|计算平台|PC（Windows/Linux，NVIDIA GPU）|运行AI模型推理、UI交互、运动规划|
|通信方式|WiFi（UDP协议）|PC与ESP32的指令传输，端口7788|
### 3.2 软件层（Software Layer，Python/PyQt5）

采用**多线程架构**，确保UI流畅性与逻辑实时性，核心模块如下：

|模块|核心类/文件|职责描述|
|---|---|---|
|主界面（UI）|`MainWindow`（[auto_sorting_system.py](auto_sorting_system.py)）|显示实时视频、控制按钮（连接/急停/运行）、参数滑块、日志输出|
|检测线程|`DetectorThread`|摄像头图像采集 → YOLOv8目标定位 → AnomalyCLIP异常推理 → 信号发射（结果/日志）|
|动作线程|`RobotActionThread`|接收目标坐标 → 路径规划 → 发送UDP指令 → 等待动作完成|
|机械臂控制|`RobotController`|封装UDP协议、逆运动学解算（IK）、S-Curve平滑加减速控制|
|AI模型推理|`AnomalyDetector`|加载YOLOv8权重、AnomalyCLIP基础模型+LoRA权重，执行前向推理|
|模型训练|`train_lora.py`|LoRA微调AnomalyCLIP，含数据加载、损失计算、权重保存|
|YOLO训练|`yolo_train.py`|训练YOLOv8定位模型，生成目标检测权重（[best.pt](best.pt)）|
## 4. 关键技术细节（Key Technical Details）

### 4.1 机械臂运动学与控制

#### 4.1.1 物理参数（Config类定义）

|参数名称|数值|含义|
|---|---|---|
|L1_BASE_H|15.0 cm|底座高度|
|L2_ARM|12.0 cm|大臂长度|
|L3_FOREARM|12.0 cm|小臂长度|
|L4_HAND_H|5.5 cm|末端执行器水平偏移|
|L4_HAND_V|4.0 cm|末端执行器垂直偏移|
|HOME_ANGLES|[0.0, 90.0, 0.0]|机械臂Home点（L型待命姿态）|
|Z_SAFE|2.0 cm|安全高度（移动时避免碰撞）|
|Z_GRAB|-0.1 cm|抓取高度（气泵接触目标）|
|BIN_X/BIN_Y/BIN_Z|8.0/-15.7/7.1 cm|废料区坐标|
#### 4.1.2 逆运动学解算（IK）

给定目标世界坐标 `(x, y, z)`，求解关节角度 `(J1, J2, J3)`：

1. **J1（底座旋转角）**： $J1 = \arctan2(y, x)$ （基于目标在XY平面的投影）

2. **坐标变换**：将末端坐标转换为手腕坐标 `(r_wrist, z_wrist)`：

    -  $r_{end} = \sqrt{x^2 + y^2}$ （末端到原点的水平距离）

    -  $r_{wrist} = r_{end} - L4_HAND_H$ （手腕水平距离）

    -  $z_{wrist} = z + L4_HAND_V$ （手腕垂直高度）

3. **J2/J3（大臂/小臂角度）**：

    - 构建三角形斜边： $Hyp = \sqrt{(r_{wrist})^2 + (z_{wrist} - L1_BASE_H)^2}$ 

    - 余弦定理求内角： $\cos\alpha = \frac{L2_ARM^2 + Hyp^2 - L3_FOREARM^2}{2 \times L2_ARM \times Hyp}$ 

    - 最终角度： $J2 = \arctan2(dz, dr) + \alpha$ ， $J3$  通过向量减法求解

#### 4.1.3 S-Curve 速度规划

为避免机械臂运动震荡与丢步，采用**梯形加减速策略**：

- 加速阶段（前25%路程）：延迟从 `START_SPEED_US（800us）` 线性减少到 `DEFAULT_SPEED_US（400us）`

- 匀速阶段（中间50%路程）：保持 `DEFAULT_SPEED_US`

- 减速阶段（后25%路程）：延迟从 `DEFAULT_SPEED_US` 线性增加到 `START_SPEED_US`

- 插补步长：`INTERPOLATION_STEP = 2.0°`（确保运动平滑）

### 4.2 坐标标定（Pixel → World）

将相机2D像素坐标 `(u, v)` 映射为机械臂3D世界坐标 `(X, Y)`，采用线性变换模型（假设相机光轴垂直于工作平面）：

- 相机内参： $cx = 1280/2$ ， $cy = 720/2$ （画面中心像素）

- 映射公式：

     $X_{robot} = OFFSET\_X - (cy - v) \times SCALE\_X$ 

     $Y_{robot} = OFFSET\_Y - (u - cx) \times SCALE\_Y$ 

- 标定参数：`SCALE_X = SCALE_Y = 0.03 mm/pixel`，`OFFSET_X = 14.7 cm`，`OFFSET_Y = 0 cm`

### 4.3 AI视觉推理流程（AI Pipeline）

#### 步骤1：目标定位（YOLOv8）

- 输入：相机实时采集帧 `I`（1280×720）

- 推理：YOLOv8加载 `best.pt` 权重，输出目标边界框 `B = (x1, y1, x2, y2)`

- 后处理：对边界框添加10像素padding，裁剪ROI（避免裁剪不全）

#### 步骤2：异常检测（AnomalyCLIP）

- 输入：裁剪后的ROI图像 `I_crop`

- 预处理：Resize(518×518) → ToTensor → Normalize（OPENAI_DATASET_MEAN/STD）

- 特征提取：

    - 图像特征 `F_img`：用于整图异常得分计算

    - Patch特征 `F_patch`：用于像素级热力图生成

- 相似度计算：

    - 异常得分： $Score = F_{img} \cdot F_{text_abnormal} / 0.07$ （Softmax后输出0~1值，超阈值判定为NG）

    - 热力图：通过 `F_patch` 与异常文本特征的局部相似度生成，经高斯滤波（ $\sigma=4$ ）平滑后叠加显示

## 5. 项目结构与关键文件（Project Structure）

|模块|文件名|功能描述|
|---|---|---|
|主程序|[auto_sorting_system.py](auto_sorting_system.py)|PC端核心程序，集成UI、YOLO/AnomalyCLIP推理、机械臂控制、线程管理|
|机械臂固件|[esp32.py](esp32.py)|ESP32底层固件，处理UDP指令、控制步进电机与气泵/电磁阀|
|YOLO训练|[yolo_train.py](yolo_train.py)|YOLOv8定位模型训练脚本，输入场景图+标注文件，输出[best.pt](best.pt)|
|AnomalyCLIP训练|[train_lora.py](train_lora.py)|LoRA微调脚本，加载ViT-L/14预训练权重，训练Prompt Learner与LoRA矩阵|
|模型核心|AnomalyCLIP_lib/|含AnomalyCLIP结构定义（DAPM、深度提示学习）、CLIP底层实现|
|辅助模块|[prompt_ensemble.py](prompt_ensemble.py)|定义Prompt Learner，生成可学习的文本提示向量|
|数据处理|[dataset.py/utils.py](dataset.py/utils.py)|数据加载、图像预处理（Resize/Normalize）、辅助函数（如归一化）|
|损失函数|[loss.py](loss.py)|定义Focal Loss和Binary Dice Loss，用于AnomalyCLIP训练|
## 6. 完整工作流程（Workflow）

### 阶段1：数据准备（Data Preparation）

#### 6.1 YOLOv8 数据

- 采集：包含目标物体的场景图（如扑克牌放在传送带上的画面）

- 标注：使用LabelImg标注目标边界框，生成 `.txt` 标签文件（YOLO格式）

- 配置：创建 `yolo_poker.yaml`，指定数据集路径、类别数等

#### 6.2 AnomalyCLIP 数据

- 采集：目标物体高清特写图（以正常样本为主，少量异常样本）

- 标注：使用 `json_to_mask.py` 生成黑底白字的缺陷掩码图（Ground Truth）

- 索引：运行 `generate_dataset_json.py` 生成 `meta.json`，记录图像与掩码的对应关系

### 阶段2：模型训练（Training）

1. 训练YOLOv8定位模型：

    ```Bash
    python yolo_train.py
    ```

    输入：场景图+标注文件 → 输出：`runs/detect/train3/weights/best.pt`

2. 微调AnomalyCLIP质检模型：

    ```Bash
    python train_lora.py --train_data_path ./data --save_path ./checkpoint_lora_pokerback
    ```

    输入：特写图+掩码+ViT-L/14预训练权重 → 输出：`best_model.pth`（Prompt权重）+ `best_lora`（LoRA权重）

### 阶段3：实时部署（Deployment）

1. 硬件连接：

    - 相机接入PC，机械臂ESP32连接WiFi（SSID：18I，密码：lichao123）

    - 确保PC与ESP32在同一局域网（默认ESP32 IP：[192.168.1.12](192.168.1.12)）

2. 启动系统：

    ```Bash
    python auto_sorting_system.py
    ```

3. 操作流程：

    - 输入ESP32 IP → 点击“连接” → 调整参数（阈值、速度）

    - 点击“开始运行” → 系统自动执行：采集→定位→质检→分拣

    - 异常品被抓取后，搬运至废料区（8.0, -15.7, 7.1）释放

## 7. 通信协议（Communication Protocol）

采用UDP无连接协议实现PC与ESP32的通信，指令格式为 `cmd,value`（纯文本字符串）。

### 7.1 核心指令定义

|Cmd ID|含义|Value说明|示例|
|---|---|---|---|
|0|电机使能|0=锁定（Enable），1=释放（Disable）|0,0|
|1/2/3|J1/J2/J3移动|相对脉冲数（正负值表示方向）|1,100（J1正转）|
|4|设置速度|脉冲延迟（微秒，越小越快）|4,600|
|5|工具控制|1=吸气，2=停止，3=释放|5,1（吸气）|
### 7.2 防丢步机制

由于UDP协议不可靠，上位机采用“阻塞式发送”策略：

1. 发送某轴运动指令后，计算运动时间  $T = |Steps| \times Delay \times 2 / 1000000.0$ 

2. 上位机睡眠  $T + 0.002s$ （预留指令传输与执行时间）

3. 依次发送下一轴指令，确保电机完成运动后再接收新指令

## 8. 操作指南（User Guide）

### 8.1 启动与连接

1. 运行 `auto_sorting_system.py`，界面加载完成后显示“就绪”。

2. 在IP输入框中填写ESP32的IP地址（默认[192.168.1.12](192.168.1.12)）。

3. 点击“连接”按钮，成功后状态栏显示“已连接：IP地址”，“去废料区”按钮激活。

### 8.2 参数调整

|参数|调整控件|推荐值|说明|
|---|---|---|---|
|异常阈值|阈值滑块（0-100）|50（0.50）|数值越大，判定越严格（减少误判NG）|
|检测间隔|间隔滑块（1-20）|5s|连续检测的时间间隔（避免频繁推理）|
|机械臂速度|速度滑块（200-2000）|600us|延迟越小速度越快（最小200us，防堵转）|
|功能开关|允许抓取/显示热力图|默认开启|关闭“允许抓取”后仅检测不分拣|
### 8.3 运行与急停

- 点击“开始运行”，按钮切换为“停止”，系统进入自动检测分拣模式。

- 检测到NG物品时，界面框选显示红色边界框+热力图，机械臂自动抓取并投放至废料区。

- 紧急情况点击红色“急停”按钮，断开机械臂连接并停止所有动作。

## 9. 已知限制与优化方向（Limitations & Future Work）

### 已知限制

1. Z轴精度依赖固定平面假设，对高度差异大的物体分拣精度不足。

2. UDP通信在网络拥堵时可能丢包，导致机械臂动作异常。

3. 热力图显示可能受光照影响，部分微小缺陷（＜5像素）难以识别。

### 优化方向

1. 引入深度相机（如Kinect），实现3D坐标精准定位，适配不同高度物体。

2. 升级通信协议为TCP或串口通信，提升指令传输可靠性。

3. 优化AnomalyCLIP的Patch特征融合策略，增强小缺陷识别能力。

4. 增加机械臂碰撞检测传感器，提升系统安全性。

5. 支持多类别缺陷分拣，扩展至更多工业场景（如电子元件、食品包装）。
