import torch
import argparse
import torch.nn.functional as F
from prompt_ensemble import AnomalyCLIP_PromptLearner
from loss import FocalLoss, BinaryDiceLoss
from utils import normalize, get_transform
from dataset import Dataset
from logger import get_logger
from tqdm import tqdm
import numpy as np
import os
import random
import AnomalyCLIP_lib
from peft import get_peft_model, LoraConfig
from sampler import get_balanced_few_shot_indices

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def find_target_modules(model):
    """自动检测视觉编码器中适合 LoRA 的目标模块"""
    target_modules = set()
    visual_model = model.visual if hasattr(model, 'visual') else model
    
    for name, module in visual_model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if "transformer" in name or "resblocks" in name:
                parts = name.split('.')
                leaf_name = parts[-1]
                if leaf_name in ['out_proj', 'c_fc', 'c_proj', 'q_proj', 'k_proj', 'v_proj', 'in_proj_weight']:
                    target_modules.add(leaf_name)
                    
    if len(target_modules) == 0:
        print("[Warning] 自动检测失败，使用默认模块")
        return ["c_fc", "c_proj", "qkv", "proj"]
    
    print(f"[Auto-detect] 找到模块: {list(target_modules)}")
    print("[Info] 使用 AnomalyCLIP 安全目标模块: ['c_fc', 'c_proj', 'qkv', 'proj']")
    return ["c_fc", "c_proj", "qkv", "proj"]

def train(args):
    logger = get_logger(args.save_path)
    preprocess, target_transform = get_transform(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    AnomalyCLIP_parameters = {
        "Prompt_length": args.n_ctx, 
        "learnabel_text_embedding_depth": args.depth, 
        "learnabel_text_embedding_length": args.t_n_ctx
    }

    # 加载模型
    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details=AnomalyCLIP_parameters)
    model.eval()

    # 先移到 CPU 避免初始化时设备不匹配
    model.to("cpu")

    # 应用 DAPM 替换（在 LoRA 包装前）
    model.visual.DAPM_replace(DPAM_layer=20)

    # 设置 LoRA
    target_modules = find_target_modules(model)
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        modules_to_save=["classifier"],
    )
    
    # 对视觉编码器应用 LoRA
    model.visual = get_peft_model(model.visual, peft_config)
    model.visual.print_trainable_parameters()  # 确认仅 LoRA 参数可训练

    # 加载训练数据
    train_data = Dataset(root=args.train_data_path, transform=preprocess, target_transform=target_transform, dataset_name=args.dataset)
    
    # 小样本平衡采样
    if args.k_shot > 0:
        indices = get_balanced_few_shot_indices(train_data, args.k_shot)
        train_sampler = torch.utils.data.SubsetRandomSampler(indices)
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=min(args.batch_size, len(indices)), sampler=train_sampler)
    else:
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    # 初始化 prompt_learner（但不训练）
    prompt_learner = AnomalyCLIP_PromptLearner(model, AnomalyCLIP_parameters)
    prompt_learner.to(device)
    model.to(device)
    
    # 加载 checkpoint（若有）
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        print(f"从 checkpoint 恢复: {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        if "prompt_learner" in checkpoint:
            prompt_learner.load_state_dict(checkpoint["prompt_learner"])
            print("已加载 prompt_learner 权重（但不参与训练）")
        if "lora_weights" in checkpoint:
            model.visual.load_state_dict(checkpoint["lora_weights"], strict=False)
            print("已加载 LoRA 权重（继续训练）")
    
    # 关键修改 1：冻结 prompt_learner 的所有参数（不更新）
    for param in prompt_learner.parameters():
        param.requires_grad = False
    prompt_learner.eval()  # 固定为评估模式，避免训练模式的副作用（如 dropout）

    # 关键修改 2：仅优化视觉编码器的 LoRA 参数（移除 prompt_learner 参数）
    params_to_optimize = [{'params': model.visual.parameters()}]
    
    optimizer = torch.optim.Adam(params_to_optimize, lr=args.learning_rate, betas=(0.5, 0.999))

    # 损失函数
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    lam = 4
    
    # 模型模式：视觉编码器 LoRA 部分可训练，其余固定；prompt_learner 已冻结
    model.eval()  # 基础模型保持评估模式，LoRA 层由 PEFT 管理训练状态
    prompt_learner.eval()  # 明确设置为评估模式
    
    for epoch in tqdm(range(args.epoch)):
        loss_list = []
        image_loss_list = []

        for items in tqdm(train_dataloader):
            image = items['img'].to(device)
            label = items['anomaly']
            gt = items['img_mask'].squeeze().to(device)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0

            # 前向传播
            image_features, patch_features = model.encode_image(image, args.features_list, DPAM_layer=20)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # prompt_learner 仅用于生成固定提示词（不更新）
            prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id=None)
            text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
            text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # 计算图像分类损失
            text_probs = image_features.unsqueeze(1) @ text_features.permute(0, 2, 1)
            text_probs = text_probs[:, 0, ...] / 0.07
            image_loss = F.cross_entropy(text_probs.squeeze(), label.long().to(device))
            image_loss_list.append(image_loss.item())
            
            # 计算相似性图损失
            similarity_map_list = []
            for idx, patch_feature in enumerate(patch_features):
                if idx >= args.feature_map_layer[0]:
                    patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)
                    similarity, _ = AnomalyCLIP_lib.compute_similarity(patch_feature, text_features[0])
                    similarity_map = AnomalyCLIP_lib.get_similarity_map(similarity[:, 1:, :], args.image_size).permute(0, 3, 1, 2)
                    similarity_map_list.append(similarity_map)

            loss = 0
            for i in range(len(similarity_map_list)):
                loss += loss_focal(similarity_map_list[i], gt)
                loss += loss_dice(similarity_map_list[i][:, 1, :, :], gt)
                loss += loss_dice(similarity_map_list[i][:, 0, :, :], 1-gt)

            loss = lam * loss
            
            # 反向传播（仅更新视觉编码器的 LoRA 参数）
            optimizer.zero_grad()
            (loss + image_loss).backward()
            optimizer.step()
            loss_list.append(loss.item())

        # 打印日志
        if (epoch + 1) % args.print_freq == 0:
            logger.info('epoch [{}/{}], loss:{:.4f}, image_loss:{:.4f}'.format(
                epoch + 1, args.epoch, np.mean(loss_list), np.mean(image_loss_list)
            ))

        # 保存 checkpoint（仅包含 LoRA 权重和固定的 prompt_learner 状态）
        if (epoch + 1) % args.save_freq == 0:
            ckp_path = os.path.join(args.save_path, 'epoch_' + str(epoch + 1) + '.pth')
            torch.save({
                "prompt_learner": prompt_learner.state_dict(),  # 仍保存但不更新
                "lora_weights": model.visual.state_dict()
            }, ckp_path)
            
            # 保存 PEFT 格式的 LoRA 权重
            model.visual.save_pretrained(os.path.join(args.save_path, f'lora_epoch_{epoch+1}'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser("AnomalyCLIP_LoRA (仅训练视觉部分)", add_help=True)
    parser.add_argument("--train_data_path", type=str, default="./data/visa", help="训练数据集路径")
    parser.add_argument("--save_path", type=str, default='./checkpoint', help='结果保存路径')
    parser.add_argument("--dataset", type=str, default='mvtec', help="数据集名称")
    parser.add_argument("--depth", type=int, default=9, help="文本嵌入深度")
    parser.add_argument("--n_ctx", type=int, default=12, help="提示词长度")
    parser.add_argument("--t_n_ctx", type=int, default=4, help="文本学习嵌入长度")
    parser.add_argument("--feature_map_layer", type=int, nargs="+", default=[0, 1, 2, 3], help="特征图层索引")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="使用的特征层")
    parser.add_argument("--epoch", type=int, default=15, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="学习率")
    parser.add_argument("--batch_size", type=int, default=8, help="批次大小")
    parser.add_argument("--image_size", type=int, default=518, help="图像尺寸")
    parser.add_argument("--print_freq", type=int, default=1, help="日志打印频率")
    parser.add_argument("--save_freq", type=int, default=1, help="模型保存频率")
    parser.add_argument("--seed", type=int, default=111, help="随机种子")
    
    # LoRA 参数
    parser.add_argument("--k_shot", type=int, default=16, help="小样本学习的样本数")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA 秩")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA  dropout")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="恢复训练的 checkpoint 路径")

    args = parser.parse_args()
    setup_seed(args.seed)
    train(args)