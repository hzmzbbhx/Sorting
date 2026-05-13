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
from sklearn.metrics import precision_recall_curve 

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def find_target_modules(model):
    """
    Automatically detect target modules for LoRA in the visual encoder.
    Focuses on Linear layers in attention and MLP blocks.
    """
    target_modules = set()
    # We only look at the visual encoder part
    visual_model = model.visual if hasattr(model, 'visual') else model
    
    for name, module in visual_model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Check if it's part of transformer blocks
            if "transformer" in name or "resblocks" in name:
                # Common names: out_proj, c_fc, c_proj
                # We use the last part of the name
                parts = name.split('.')
                leaf_name = parts[-1]
                
                # Filter for likely candidates
                if leaf_name in ['out_proj', 'c_fc', 'c_proj', 'q_proj', 'k_proj', 'v_proj', 'in_proj_weight']:
                    target_modules.add(leaf_name)
                    
    if len(target_modules) == 0:
        print("[Warning] Auto-detection failed, using default candidates.")
        # Use 'proj' and 'qkv' for custom Attention (DAPM), 'c_fc' and 'c_proj' for MLP.
        # Avoid 'out_proj' to prevent issues with nn.MultiheadAttention in early layers.
        return ["c_fc", "c_proj", "qkv", "proj"]
    
    # Filter out 'out_proj' if it causes issues, or rely on the fact that we prefer 'proj' for custom attention
    # If auto-detect found 'out_proj', it might be from MultiheadAttention.
    # Let's explicitly exclude 'out_proj' and ensure 'proj'/'qkv' are included if present.
    # Actually, let's just override with the known safe list for AnomalyCLIP
    print(f"[Auto-detect] Found modules: {list(target_modules)}")
    print("[Info] Using safe target modules for AnomalyCLIP: ['c_fc', 'c_proj', 'qkv', 'proj']")
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

    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details=AnomalyCLIP_parameters)
    model.eval()

    # Move to CPU first to avoid device mismatch issues during initialization
    model.to("cpu")

    # Apply DAPM replacement BEFORE LoRA wrapping
    # This ensures we are wrapping the correct Attention modules
    model.visual.DAPM_replace(DPAM_layer=20)

    # Setup LoRA
    target_modules = find_target_modules(model)
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        modules_to_save=["classifier"], # If there's a classifier head, but CLIP usually doesn't have one in this context
    )
    
    # Apply LoRA to the visual encoder
    model.visual = get_peft_model(model.visual, peft_config)
    model.visual.print_trainable_parameters()

    train_data = Dataset(root=args.train_data_path, transform=preprocess, target_transform=target_transform, dataset_name=args.dataset, mode='train')
    
    # Use balanced sampling for few-shot
    # If k_shot is provided, we sample indices
    if args.k_shot > 0:
        indices = get_balanced_few_shot_indices(train_data, args.k_shot)
        train_sampler = torch.utils.data.SubsetRandomSampler(indices)
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=min(args.batch_size, len(indices)), sampler=train_sampler)
    else:
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    prompt_learner = AnomalyCLIP_PromptLearner(model, AnomalyCLIP_parameters)
    prompt_learner.to(device)
    model.to(device)
    
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        print(f"Resuming from checkpoint: {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        if "prompt_learner" in checkpoint:
            prompt_learner.load_state_dict(checkpoint["prompt_learner"])
            print("Loaded prompt_learner weights.")
        if "lora_weights" in checkpoint:
            model.visual.load_state_dict(checkpoint["lora_weights"], strict=False)
            print("Loaded LoRA weights.")
    
    # Freeze non-LoRA parameters in visual encoder (get_peft_model does this mostly, but let's be sure)
    # And decide if we want to train prompt_learner. 
    # Usually for few-shot adaptation we might want to train both or just LoRA.
    # Let's train both.
    
    params_to_optimize = [
        {'params': model.visual.parameters()},
        {'params': prompt_learner.parameters()}
    ]
    
    optimizer = torch.optim.Adam(params_to_optimize, lr=args.learning_rate, betas=(0.5, 0.999))
    
    # 学习率自动调整：当训练损失停止改善时降低学习率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',  # 监控指标最小化
        factor=0.5,  # 学习率调整因子
        patience=3,  # 多少个epoch无改善后调整
        verbose=True,
        min_lr=1e-6  # 最小学习率
    )

    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    lam = 4
    
    model.eval() # CLIP visual encoder usually stays in eval mode (BatchNorm etc) except LoRA layers
    # But PEFT handles LoRA layers training mode.
    # prompt_learner should be in train mode.
    prompt_learner.train()
    
    # 早停参数
    best_loss = float('inf')
    patience = 12  # 容忍多少个epoch没有改善
    counter = 0   # 计数器
    
    for epoch in tqdm(range(args.epoch)):
        loss_list = []
        image_loss_list = []

# === 新增：初始化用于存储整个epoch数据的列表 ===
        epoch_gt_list = []  # 存放真实标签
        epoch_pr_list = []  # 存放预测的异常概率

        for items in tqdm(train_dataloader):
            image = items['img'].to(device)
            label = items['anomaly']
            gt = items['img_mask'].squeeze().to(device)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0

            # Forward pass
            # Note: model.encode_image might need adjustment if LoRA changes the signature or return type
            # But PEFT wraps the module so it should be fine if it forwards calls.
            # However, AnomalyCLIP_lib.load returns a custom CLIP model.
            # We wrapped model.visual.
            
            image_features, patch_features = model.encode_image(image, args.features_list, DPAM_layer=20)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id=None)
            text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
            text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # 原始代码计算 logits 用于 loss
            text_probs_logits = image_features.unsqueeze(1) @ text_features.permute(0, 2, 1)
            text_probs_logits = text_probs_logits[:, 0, ...] / 0.07
            
          
            image_loss = F.cross_entropy(text_probs_logits.squeeze(), label.long().to(device))
            image_loss_list.append(image_loss.item())

            # === 新增：计算并收集概率值用于寻找阈值 ===
            # 对 logits 进行 softmax 得到概率，取 index=1 (异常类的概率)
            probs = torch.softmax(text_probs_logits.squeeze(), dim=-1)[:, 1]
            epoch_gt_list.extend(label.cpu().numpy())
            epoch_pr_list.extend(probs.detach().cpu().numpy())
            
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
            
            optimizer.zero_grad()
            (loss + image_loss).backward()
            optimizer.step()
            loss_list.append(loss.item())

        avg_loss = np.mean(loss_list)
        avg_image_loss = np.mean(image_loss_list)
        
        # === 新增：计算最佳 F1 阈值 ===
        # 注意：如果训练集中全是正常样本(label全为0)，会报错，需要加个判断
        if len(np.unique(epoch_gt_list)) > 1:
            precision, recall, thresholds = precision_recall_curve(epoch_gt_list, epoch_pr_list)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            best_idx = np.argmax(f1_scores)
            best_threshold = thresholds[best_idx]
            best_f1 = f1_scores[best_idx]
            threshold_msg = f", Best Thres: {best_threshold:.4f}, Best F1: {best_f1:.4f}"
        else:
            best_threshold = 0.5 # 默认值
            threshold_msg = ", (Only one class in batch, cannot calc threshold)"
        # ============================
        
        if (epoch + 1) % args.print_freq == 0:
            # 修改 log 信息，把阈值打印出来
            logger.info('epoch [{}/{}], loss:{:.4f}, image_loss:{:.4f}{}'.format(
                epoch + 1, args.epoch, avg_loss, avg_image_loss, threshold_msg))
        
        scheduler.step(avg_loss)
        
        # 早停检查：监控平均训练损失
        if avg_image_loss < best_loss:
            best_loss = avg_image_loss
            counter = 0  # 重置计数器
            # 可以保存最佳模型
            best_ckp_path = os.path.join(args.save_path, 'best_model.pth')
            torch.save({
                "prompt_learner": prompt_learner.state_dict(),
                "lora_weights": model.visual.state_dict()
            }, best_ckp_path)
            model.visual.save_pretrained(os.path.join(args.save_path, 'best_lora'))
        else:
            counter += 1
            if counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break  # 退出训练循环
        
        if (epoch + 1) % args.save_freq == 0:
            ckp_path = os.path.join(args.save_path, 'epoch_' + str(epoch + 1) + '.pth')
            # Save both prompt learner and LoRA weights
            torch.save({
                "prompt_learner": prompt_learner.state_dict(),
                "lora_weights": model.visual.state_dict() # Or use model.visual.save_pretrained
            }, ckp_path)
            
            # Also save PEFT config/weights properly
            model.visual.save_pretrained(os.path.join(args.save_path, f'lora_epoch_{epoch+1}'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser("AnomalyCLIP_LoRA", add_help=True)
    parser.add_argument("--train_data_path", type=str, default="./data/visa", help="train dataset path")
    parser.add_argument("--save_path", type=str, default='./checkpoint', help='path to save results')
    parser.add_argument("--dataset", type=str, default='mvtec', help="train dataset name")
    parser.add_argument("--depth", type=int, default=9, help="image size")
    parser.add_argument("--n_ctx", type=int, default=12, help="zero shot")
    parser.add_argument("--t_n_ctx", type=int, default=4, help="zero shot")
    parser.add_argument("--feature_map_layer", type=int, nargs="+", default=[0, 1, 2, 3], help="zero shot")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
    parser.add_argument("--epoch", type=int, default=15, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--print_freq", type=int, default=1, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=1, help="save frequency")
    parser.add_argument("--seed", type=int, default=111, help="random seed")
    
    # LoRA args
    parser.add_argument("--k_shot", type=int, default=16, help="Number of shots for few-shot learning")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA r")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="path to checkpoint to resume")

    args = parser.parse_args()
    setup_seed(args.seed)
    train(args)