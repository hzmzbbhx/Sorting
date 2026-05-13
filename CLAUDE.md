# CLAUDE.md（中文版）

> 本文档用于指导 Claude Code 理解本项目的架构、算法细节和常用操作。

## 项目概述

本项目基于 **AnomalyCLIP**（ICLR 2024）——一种零样本异常检测（ZSAD）框架。核心思想是学习 **object-agnostic（对象无关）** 的文本提示词，捕捉通用的"正常/异常"概念，而非特定物体的语义，从而实现跨域泛化。

### 我的改进工作

在原版 AnomalyCLIP 基础上，我做了以下扩展：

1. **LoRA 低秩自适应微调**：在 ViT 视觉编码器的 Attention/MLP 层旁路插入低秩矩阵 A 和 B，用极少样本（10-30 张图）快速适配特定工业场景，解决原版零样本在某些场景精度不足的问题。
2. **YOLOv8 + AnomalyCLIP 两级分拣系统**：YOLOv8 定位目标 → 裁剪 ROI → LoRA 微调后的 AnomalyCLIP 做像素级异常检测 → 机械臂自动分拣。
3. **多策略对比实验**：视觉端 LoRA 单独微调 vs Prompt+LoRA 联合微调，以及学习率/epoch 的消融实验。
4. **知识蒸馏边缘部署**：以 LoRA 微调后的 AnomalyCLIP（ViT-L，304M）为教师，蒸馏出轻量级 ViT-B 学生模型（86M，3.5× 压缩），用于工业边缘端实时推理。详见 [`docs/知识蒸馏方案.md`](docs/知识蒸馏方案.md)。

## 环境与依赖

- PyTorch 2.0.0+
- CUDA GPU（推荐 RTX 3090 24GB）
- `pip install torch torchvision peft ultralytics scipy tabulate scikit-learn matplotlib`
- `pip install labelme`（用于标注转掩码，可选）

## 项目目录结构

```
AnomalyCLIP/
├── AnomalyCLIP_lib/          # 核心框架（原作者）
├── training/                  # 训练脚本
│   ├── train.py              # 原作者 Prompt Learning 训练
│   ├── train_lora.py         # LoRA + Prompt 联合微调（主要使用）
│   ├── train_only_visual.py  # 仅视觉端 LoRA 微调（消融实验）
│   ├── threshold.py          # 自动寻找最佳分类阈值
│   ├── distill.py            # 知识蒸馏训练（教师→学生）
│   └── yolo_train.py          # 自动寻找最佳分类阈值
├── inference/                 # 推理/测试脚本
│   ├── test.py               # 原作者标准测试
│   ├── test_lora.py          # LoRA 模型测试（主要使用）
│   ├── test_distill.py       # 蒸馏学生模型测试
│   └── test_one_example.py   # 单张图片快速推理
├── sorting_system/            # YOLO + AnomalyCLIP 智能分拣系统
│   ├── auto_sorting_system.py # 完整分拣系统（最终版）
│   ├── yolo_train.py         # YOLOv8 定位模型训练
│   ├── monitor.py            # 早期纯检测 UI
│   ├── monitor_lora.py       # LoRA 版检测 UI
│   ├── monior_complete.py    # 增强版检测 UI
│   └── yolo_monitior.py      # YOLO + AnomalyCLIP 早期原型
├── robotics/                  # 机械臂控制
│   ├── robot_arm_only.py     # 独立上位机控制程序
│   └── esp.py                # ESP32 底层固件
├── data_tools/                # 数据处理工具
│   ├── generate_dataset_json/ # 各数据集 meta.json 生成
│   └── mask_product.py       # LabelMe JSON → 掩码 PNG
├── scripts/                   # 启动脚本
│   ├── train.bat  / train.sh (原)
│   ├── test.bat  / test.sh  (原)
│   ├── test_one_example.sh
│   ├── run_lora.bat          # LoRA 训练启动
│   ├── run_sdd_lora.bat       # SDD LoRA 教师训练
│   ├── run_distill_sdd.bat    # SDD 知识蒸馏
│   └── run_test_distill_sdd.bat # 蒸馏对比测试
├── docs/                      # 文档
│   ├── 复现流程说明.md
│   ├── 项目文件说明.md
│   ├── ppt第一次汇报.md
│   ├── 数据分析.txt
│   └── 机械臂智能分拣系统技术文档...md
├── tools/                     # 辅助工具
├── prompt_ensemble.py         # Prompt Learner（根目录共享模块）
├── loss.py                    # Focal Loss + Dice Loss
├── metrics.py                 # 评估指标
├── dataset.py                 # 数据加载器
├── utils.py                   # 图像预处理
├── visualization.py           # 热力图可视化
├── logger.py                  # 日志
├── sampler.py                 # 少样本平衡采样
├── .gitignore
├── CLAUDE.md                  # 本文档
└── README.md                  # 原作者 README
```

## 核心算法架构（深入理解）

### 1. 基础模型：CLIP ViT-L/14@336px

- 视觉编码器：ViT-Large，输入分辨率 336×336（patch size=14，共 24×24=576 个 patch token）
- 文本编码器：12 层 Transformer
- 共享嵌入维度：768
- 预训练权重路径：`ViT-L-14-336px.pt`（OpenAI 官方）

### 2. DPAM（Dual-Path Attention Mechanism，双路径注意力机制）

**注意**：DPAM 是 Dual-Path Attention Mechanism 的缩写。其核心是在 ViT 的深层（第 20 层起，共 24 层）替换标准 self-attention，同时计算两条路径：

- **路径 1（原始）**：标准 self-attention，`softmax(Q·K^T/√d) · V`
- **路径 2（新增）**：v-v self-attention，`softmax(V·V^T/√d) · V`

其中 Q、K、V 共用同一个线性投影（`in_proj_weight`）。路径 2 的核心在于计算 `V·V^T`（类似 Gram 矩阵），强制模型关注特征间的**纹理相关性**，而非仅依赖空间语义。这使得模型对划痕、异色、污渍等破坏纹理连续性的微小缺陷变得敏感。

**代码位置**：`AnomalyCLIP_lib/AnomalyCLIP.py` 中的 `Attention` 类。

**重要**：DAPM 替换必须在 LoRA 包装之前执行（`model.visual.DAPM_replace(DPAM_layer=20)` 在 `get_peft_model()` 之前）。

### 3. Deep Prompt Learning（深度提示学习）

`AnomalyCLIP_PromptLearner`（`prompt_ensemble.py`）学习 object-agnostic 的文本嵌入：

- **输入层可学习 token**：构造 `[ctx_1 ... ctx_n][object]` 和 `[ctx_1 ... ctx_n][damaged object]` 两种模板，其中 `ctx` 向量可学习，`[object]` 是从原始 CLIP token 嵌入中获取的固定 token。
- **深层注入 token**（`compound_prompts_text`）：在文本 Transformer 的每一层（由 `--depth` 控制，默认 9 层）注入可学习的 token 向量，让模型在特征提取的全流程中持续强化"正常/异常"的概念认知。
- 前向返回：`(prompts, tokenized_prompts, compound_prompts_text)`，消费于 `model.encode_text_learn()`。

**参数数量**（典型的 depth=9, n_ctx=12, t_n_ctx=4 配置）：
- ctx 向量：12 × 768 = 9,216
- compound_prompts：9 层 × 4 × 768 = 27,648
- 总计约 37K 可训练参数

### 4. LoRA（Low-Rank Adaptation）低秩自适应

在冻结的 CLIP 原始参数旁插入低秩矩阵 A（d×r）和 B（r×d），仅训练 A 和 B：

```
W_new = W_frozen + (α/r) · B · A
```

**目标模块**（适配 AnomalyCLIP 结构）：
- `c_fc`、`c_proj`：MLP 层的全连接
- `qkv`、`proj`：DAPM 自定义 Attention 的投影层
- 避开 `out_proj`（来自 nn.MultiheadAttention 的低层结构，会导致 shape 不匹配错误）

**参数配置**：
- `lora_r`（秩，rank）：默认 8-16，越小参数量越少
- `lora_alpha`（缩放因子）：默认 16，实际缩放 = α/r
- `lora_dropout`：默认 0.1

**训练参数**（以 Pokerback 为例，k_shot=26）：
- 总可训练参数 ≈ 1.07M（LoRA ≈ 1.04M + PromptLearner ≈ 37K）
- ViT-L/14 总参数 ≈ 304M，训练参数仅占 0.35%

### 5. 推理流程（完整链路）

```
图像输入 (H×W)
  │
  ▼
ViT-L/14 编码（DAPM 生效，第 20 层起双路径）
  │
  ├─ [CLS] token → image_feature (全局特征, 1×768)
  │
  └─ patch tokens (第 6, 12, 18, 24 层) → patch_features (局部特征)
  │
  ▼
文本特征提取（PromptLearner 生成 prompt → 文本编码器）
  │
  ├─ text_normal (1×768)
  └─ text_abnormal (1×768)
  stack → text_features (1×2×768)
  │
  ▼
图像级异常得分:
  score = softmax(image_feature @ text_features^T / 0.07)[abnormal]
  (temperature τ=0.07 是 CLIP 训练时的 logit_scale 的倒数)
  │
  ▼
像素级异常热力图:
  for each patch_feature in [layer6, layer12, layer18, layer24]:
      similarity = patch_feature @ text_features^T  → [N_patches, 2]
      reshape → spatial similarity map
  anomaly_map = mean(all layers)[ abnormal ]
  gaussian_filter(anomaly_map, σ=4)
```

### 6. 损失函数

训练 PromptLearner 和 LoRA 使用三重损失：

| 损失 | 公式 | 作用 |
|------|------|------|
| **Image Cross Entropy** | `CE(softmax(image_feat @ text_feat / 0.07), label)` | 确保整图分类（OK/NG）正确 |
| **Focal Loss** | `-α(1-p_t)^γ log(p_t)`，α=0.25, γ=2 | 解决像素不平衡（缺陷像素 << 正常像素） |
| **Binary Dice Loss** | `1 - (2\|P∩G\|+ε) / (\|P\|+\|G\|+ε)` | 优化预测掩码与 GT 掩码的 IoU 重叠度 |

总损失：`λ × (FocalLoss + DiceLoss) + CrossEntropy`，λ=4

### 7. 知识蒸馏（Knowledge Distillation）

将 AnomalyCLIP 从 ViT-L 教师（304M）压缩到 ViT-B 学生（86M），用于边缘端部署。

#### 架构对比

| 组件 | 教师 | 学生 |
|------|------|------|
| 基础模型 | CLIP ViT-L/14@336px | CLIP ViT-B/16@224~336px |
| 层数 / 宽度 / 头数 | 24 / 1024 / 16 | 12 / 768 / 12 |
| DAPM 层 | 20（后19层双路径） | 12（全部11层双路径） |
| 参数量 | ~304M + LoRA ~1M | ~86M（全参数，无 LoRA） |
| Patch 网格 | 37×37 (518/14) | 14×14 (224/16) 或 21×21 (336/16) |
| 文本编码器 | 冻结，共享 | 共享教师的文本编码器 + PromptLearner |

#### 共享策略

文本编码器 + PromptLearner 仅 ~37K 参数，且 CLIP 多模态空间统一投影到 768 维，因此教师和学生共享同一套文本特征，**仅蒸馏视觉编码器**。

#### 蒸馏损失设计

| 损失 | 类型 | 公式 | 作用 |
|------|------|------|------|
| CLS 余弦相似度 | 全局特征对齐 | `1 - cos(student_cls, teacher_cls)` | 对齐图像级特征方向 |
| 相似度图 KL 散度 | 局部分布匹配 | `KL(teacher_logit/T ‖ student_logit/T)` | 迁移逐 patch 异常判别知识 |
| 任务损失 | GT 辅助正则 | `FocalLoss + DiceLoss + CrossEntropy` | 防止过拟合教师误差 |

总损失：`λ_cls × L_cls + λ_sim × L_sim + λ_task × L_task`

推荐权重：λ_cls=5.0, λ_sim=30.0, λ_task=1.0, T=2.0

#### 关键实现细节

1. **蒸馏对象**：使用原始 logit（`patch_feat @ text_feat^T / 0.07`，未做 softmax）做 KL 蒸馏，避免 `compute_similarity` 自带 softmax 导致的二次 softmax 压平问题。
2. **温度缩放**：T=2.0 适度软化教师分布，保留判别信息的同时提供更丰富的梯度。
3. **DAPM 全层替代**：学生 DAPM_layer=12（全部 12 层），弥补低分辨率和浅层数对异常敏感度的损失。
4. **位置嵌入插值**：迁移 ViT-B/16 预训练权重时，对 positional_embedding 做 bilinear 插值以适配不同分辨率。

#### 蒸馏训练命令

```bash
# 推荐：直接运行批处理脚本
scripts\run_distill_sdd.bat

# 或手动指定参数
python training/distill.py \
    --train_data_path ./SDD --save_path ./checkpoint_distill_sdd \
    --dataset SDD \
    --teacher_lora_path ./checkpoint_lora_sdd/best_lora \
    --checkpoint_path ./checkpoint_lora_sdd/best_model.pth \
    --image_size 336 --batch_size 2 --epoch 20 \
    --student_dpam_layer 12 \
    --lambda_cls 5.0 --lambda_sim 30.0 --lambda_task 1.0 \
    --distill_temperature 2.0
```

#### 蒸馏模型测试

```bash
# 推荐：直接运行批处理脚本（含教师对比）
scripts\run_test_distill_sdd.bat

# 或单独测试学生
python inference/test_distill.py \
    --data_path ./SDD --save_path ./results_distill_sdd/student \
    --checkpoint_path ./checkpoint_lora_sdd/best_model.pth \
    --student_checkpoint ./checkpoint_distill_sdd/student_best.pth \
    --dataset SDD --image_size 336 \
    --student_dpam_layer 12 --metrics image-pixel-level
```

## 常用命令

### 1. 生成数据集 JSON（训练/测试前必须运行）

```bash
cd data_tools/generate_dataset_json
python mvtec.py      # MVTec AD
python visa.py        # VisA
python SDD.py         # SDD
python DTD.py         # DTD-Synthetic
python Pokerback.py   # 扑克牌背面数据集（自定义）
# ...
```

### 2. 原版 Prompt Learning 训练（baseline）

```bash
# Windows
scripts/train.bat

# Linux
CUDA_VISIBLE_DEVICES=0 python training/train.py \
    --dataset visa --train_data_path ./data/visa \
    --save_path ./checkpoints/ --features_list 24 \
    --image_size 518 --batch_size 8 --epoch 15 \
    --depth 9 --n_ctx 12 --t_n_ctx 4
```

### 3. LoRA 少样本微调训练

```bash
# Windows（推荐）
scripts/run_lora.bat

# Linux / 手动
python training/train_lora.py \
    --train_data_path ./Pokerback \
    --save_path ./checkpoint_lora_pokerback \
    --checkpoint_path ./checkpoints/9_12_4_multiscale/epoch_15.pth \
    --dataset Pokerback \
    --k_shot 26 --lora_r 8 --lora_alpha 16 \
    --epoch 30 --learning_rate 0.001
```

### 4. LoRA 模型测试

```bash
python inference/test_lora.py \
    --data_path ./Pokerback \
    --save_path ./results_lora/ \
    --checkpoint_path ./checkpoint_lora_pokerback/best_model.pth \
    --lora_path ./checkpoint_lora_pokerback/best_lora \
    --dataset Pokerback \
    --metrics image-pixel-level
```

### 5. 完整分拣系统运行

```bash
python sorting_system/auto_sorting_system.py
```

### 6. YOLOv8 定位模型训练

```bash
python sorting_system/yolo_train.py
```

## 实验结果（已验证）

### SDD 数据集（低基线、高增益场景）

| 策略 | Pixel AUROC | Pixel AUPRO | Image AUROC | Image AP |
|------|:-----------:|:-----------:|:-----------:|:--------:|
| 未微调（零样本） | 90.1 | 62.9 | 82.7 | 73.8 |
| 仅 LoRA 视觉端 | 90.0 | 65.8 | 82.9 | 74.3 |
| 联合微调（lr=0.001, epoch=30） | 89.2 | 64.7 | 84.7 | 79.0 |
| 联合微调（lr=0.01, epoch=30） | 91.8 | 84.2 | 87.9 | 86.8 |
| **联合微调（lr=0.01, epoch=70）** | **93.4** | **84.9** | **87.8** | **86.7** |

> 关键发现：lr=0.001 时 loss 原地打转，提升到 0.01 后模型才开始真正学习。epoch=30 时 Pixel AUPRO 已从 62.9 跃至 84.2（+21.3），epoch=70 时达到最佳。仅调视觉端效果甚微（Pixel AUPRO 仅 +2.9），说明文本端 Prompt Learning 不可缺失。

### DTD 数据集（高基线、低增益场景）

| 策略 | Pixel AUROC | Pixel AUPRO | Image AUROC | Image AP |
|------|:-----------:|:-----------:|:-----------:|:--------:|
| 未微调（零样本） | 97.6 | 87.7 | 94.5 | 97.7 |
| 联合微调 | 96.8 | 90.9 | 95.2 | 98.0 |
| **联合微调（k-shot=200）** | **98.3** | **94.6** | **96.2** | **98.3** |

> DTD 基线本身就高（零样本已有 97.7% Image AP），微调提升有限（Pixel AUPRO +6.9），说明模型对纹理数据已有较强的泛化能力。k-shot 增大到 200 后有进一步增益。

### 跨域泛化实验

| 模型 | 测试数据 | Image AUROC | Image AP |
|------|----------|:-----------:|:--------:|
| 零样本（未微调） | BrainMRI | 90.3 | 92.2 |
| SDD LoRA（epoch=30） | BrainMRI | **96.2** | **97.2** |

> 趣味发现：用 SDD 少样本微调后的 LoRA 权重直接测 BrainMRI，效果反而远超零样本基线（Image AUROC +5.9, Image AP +5.0）。推测原因是 SDD 的缺陷形态（换向器划痕/污渍）与 BrainMRI 的肿瘤异常在纹理特征上存在相似性，LoRA 微调强化了模型对这种"不规则纹理异常"的检测能力，而 Prompt Learning 保持了语义端的通用性。"

### SDD 知识蒸馏实验（教师→学生）

| 版本 | 配置 | Pixel AUROC | Pixel AUPRO | Image AUROC | Image AP |
|------|------|:-----------:|:-----------:|:-----------:|:--------:|
| 教师 | ViT-L/14@518 + LoRA + DAPM=20 | 90.7 | 81.4 | 87.5 | 86.2 |
| v1 蒸馏 | MSE, 权重失衡, DAPM=8 | 78.9 | 34.9 | 71.6 | 53.8 |
| v3 蒸馏 | 余弦+KL, 权重修正, DAPM=8 | 77.7 | 51.9 | 75.7 | 56.7 |
| v4 蒸馏 | + DAPM=12 | 86.0 | 61.3 | 74.1 | 55.4 |
| **v5 蒸馏** | **+ 336px 分辨率** | **87.2** | **62.3** | 73.6 | **61.2** |
| **v1→v5 提升** | | **+8.3** | **+27.4** | **+2.0** | **+7.4** |

> 关键发现：(1) 余弦相似度损失远优于 MSE——特征已 L2 归一化，MSE 无法捕捉方向差异；(2) KL 散度必须使用原始 logit（未 softmax），二次 softmax 会将分布压平到均匀使损失失效；(3) DAPM 全层替代（DAPM_layer=12）对像素指标提升最显著（AUPRO +9.4）；(4) 学生 336px 分辨率优于 224px（+1.2 AUROC, +5.8 AP）；(5) 学生模型以 3.5× 参数压缩（304M→86M），像素 AUROC 仅差 3.5 点，基本满足边缘部署需求。

详细数据见 [`results_distill_sdd/对比汇总.md`](results_distill_sdd/对比汇总.md)。

## 技术要点提醒

1. **DAPM 替换时序**：必须在 LoRA 包装之前执行 `model.visual.DAPM_replace(DPAM_layer=20)`，否则会包装到错误的 Attention 模块。
2. **模型加载路径**：`AnomalyCLIP_lib/model_load.py` 第 38 行硬编码了缓存路径 `/remote-home/iot_zhouqihang/root/.cache/clip`，本地使用需修改。
3. **设备迁移顺序**：先 `model.to("cpu")` → DAPM_replace → LoRA 包装 → `model.to(device)`，避免设备不匹配导致的初始化错误。
4. **PEFT 权重保存**：使用 `model.visual.save_pretrained(path)` 保存 PEFT 格式；`torch.save(state_dict, path)` 保存完整 state_dict。推荐两者同时保存。
5. **学习率选择**：对于 SDD 等数据量较小的场景，lr=0.01 比默认的 0.001 效果明显更好。如果 loss 不下降，尝试提高 lr。
6. **少样本采样**：`sampler.py` 会优先选择带掩码的异常样本，确保 1:1 的类别平衡。
7. **蒸馏-教师前向必须 no_grad**：蒸馏训练中教师前向需包裹 `with torch.no_grad()`，冻结教师参数避免显存浪费。
8. **ViT 前向 `@torch.no_grad()` 已移除**：原版 `AnomalyCLIP_lib/AnomalyCLIP.py` 中 `VisionTransformer.forward` 的 `@torch.no_grad()` 已被移除，否则学生模型梯度无法回传。
9. **DAPM inplace `+=` 已修复**：原版 `ResidualAttentionBlock.forward` 中 `x_ori += x_ori_res` / `x += x_res` 已改为 `x = x + x_res`（out-of-place），否则梯度计算报错。
10. **蒸馏-KL 必须用原始 logit**：`compute_similarity()` 内部做了 softmax，蒸馏时直接使用原始 logit（`patch_feat @ text_feat^T / 0.07`）做 KL，避免二次 softmax 压平分布。
11. **位置嵌入 resize**：学生迁移 ViT-B/16 预训练权重时，若 input_resolution≠224，需对 `positional_embedding` 做 bilinear 插值。见 `distill.py` 中 `build_student_visual()`。

## 已知限制

1. Z 轴精度依赖固定平面假设，对高度差异大的物体分拣精度不足。
2. UDP 通信可能丢包，高负载时机械臂动作可能异常。
3. 部分微小缺陷（< 5 像素）检测能力有限，受限于 ViT patch 分辨率。
4. 当前仅支持二分类（OK/NG），不支持多类别缺陷细分。
