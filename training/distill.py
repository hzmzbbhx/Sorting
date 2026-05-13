"""
Knowledge Distillation: AnomalyCLIP ViT-L (Teacher) -> ViT-B (Student)

Teacher: AnomalyCLIP ViT-L/14@336px + LoRA + DAPM layer 20 (frozen)
Student: AnomalyCLIP VisionTransformer ViT-B/16@224 + DAPM layer 8 (trainable)

Distillation losses:
  1. CLS Feature MSE (teacher vs student global features)
  2. Anomaly Map MSE (teacher vs student pixel-level predictions)
  3. Task Loss (Focal + Dice + CE) against ground truth
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import sys
import numpy as np
import random
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import AnomalyCLIP_lib
from AnomalyCLIP_lib.AnomalyCLIP import VisionTransformer
from prompt_ensemble import AnomalyCLIP_PromptLearner
from loss import FocalLoss, BinaryDiceLoss
from utils import get_transform
from dataset import Dataset
from sampler import get_balanced_few_shot_indices
from logger import get_logger
from peft import PeftModel


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


VIT_B_16_URL = "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt"


def download_vit_b16(cache_dir="./pretrained"):
    """Download ViT-B/16 from OpenAI CDN and return the JIT model."""
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, "ViT-B-16.pt")
    if not os.path.exists(fname):
        print(f"[Download] Fetching ViT-B/16 from OpenAI CDN...")
        torch.hub.download_url_to_file(VIT_B_16_URL, fname, progress=True)
        print(f"[Download] Saved to {fname}")
    return torch.jit.load(fname, map_location="cpu")


def build_student_visual(device, output_dim=768, input_resolution=224):
    """
    Build student visual encoder: AnomalyCLIP VisionTransformer with ViT-B/16 dims.
    Initialize weights from OpenAI pretrained ViT-B/16.
    input_resolution: 224 (14x14 patches) or 336 (21x21 patches) for finer localization.
    """
    student = VisionTransformer(
        input_resolution=input_resolution,
        patch_size=16,
        width=768,
        layers=12,
        heads=12,
        output_dim=output_dim
    )

    try:
        jit_model = download_vit_b16()
        pretrained_sd = jit_model.visual.state_dict()
        student_sd = student.state_dict()

        loaded = 0
        skipped = []
        for key in student_sd:
            if key in pretrained_sd:
                if student_sd[key].shape == pretrained_sd[key].shape:
                    student_sd[key] = pretrained_sd[key].clone()
                    loaded += 1
                else:
                    skipped.append(f"{key}: stu{tuple(student_sd[key].shape)} vs pre{tuple(pretrained_sd[key].shape)}")

        # Handle proj shape mismatch: original ViT-B/16 proj is [768, 512], ours is [768, 768]
        if 'proj' in pretrained_sd and 'proj' in student_sd:
            orig = pretrained_sd['proj']  # [768, 512]
            tgt = student_sd['proj']       # [768, 768]
            copy_cols = min(orig.shape[1], tgt.shape[1])
            student_sd['proj'][:, :copy_cols] = orig[:, :copy_cols].clone()
            print(f"[Student] proj: copied first {copy_cols} cols from pretrained [768,512] to [768,{output_dim}]")

        student.load_state_dict(student_sd)
        print(f"[Student] Loaded {loaded} params from pretrained ViT-B/16")
        if skipped:
            print(f"[Student] Shape mismatches (random init): {skipped}")

        del jit_model

        # Resize positional embedding if input_resolution differs from pretrained 224
        if input_resolution != 224:
            old_grid = 14  # 224 / 16
            new_grid = input_resolution // 16
            cls_token = student.positional_embedding.data[:1, :]  # [1, 768]
            patch_pos = student.positional_embedding.data[1:, :]  # [196, 768]
            patch_pos = patch_pos.reshape(old_grid, old_grid, 768).permute(2, 0, 1).unsqueeze(0)  # [1, 768, 14, 14]
            new_patch_pos = F.interpolate(patch_pos, (new_grid, new_grid), mode='bilinear', align_corners=False)
            new_patch_pos = new_patch_pos.squeeze(0).permute(1, 2, 0).reshape(-1, 768)  # [N, 768]
            student.positional_embedding = nn.Parameter(torch.cat([cls_token, new_patch_pos], dim=0))
            print(f"[Student] Resized positional embedding: [{old_grid}x{old_grid}] -> [{new_grid}x{new_grid}]")

    except Exception as e:
        print(f"[Student] Pretrained weight loading failed: {e}")
        print("[Student] Using random initialization (training from scratch)")

    student.to(device)
    return student


def train_distill(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = get_logger(args.save_path)
    preprocess, target_transform = get_transform(args)

    # ========== Teacher Setup (Frozen) ==========
    logger.info("=== Loading Teacher: AnomalyCLIP ViT-L/14@336px + LoRA ===")
    AnomalyCLIP_parameters = {
        "Prompt_length": args.n_ctx,
        "learnabel_text_embedding_depth": args.depth,
        "learnabel_text_embedding_length": args.t_n_ctx
    }

    teacher, _ = AnomalyCLIP_lib.load(
        "ViT-L/14@336px", device=device, design_details=AnomalyCLIP_parameters)
    teacher.to("cpu")
    teacher.visual.DAPM_replace(DPAM_layer=20)
    teacher.visual = PeftModel.from_pretrained(teacher.visual, args.teacher_lora_path)
    teacher.eval()
    teacher.to(device)

    # ========== PromptLearner (Shared, Frozen) ==========
    prompt_learner = AnomalyCLIP_PromptLearner(teacher, AnomalyCLIP_parameters)
    if os.path.exists(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        if "prompt_learner" in checkpoint:
            prompt_learner.load_state_dict(checkpoint["prompt_learner"])
        else:
            prompt_learner.load_state_dict(checkpoint)
        logger.info(f"Loaded PromptLearner from {args.checkpoint_path}")
    prompt_learner.to(device)
    prompt_learner.eval()

    # Precompute text features (frozen, shared by teacher and student)
    logger.info("=== Precomputing Text Features ===")
    with torch.no_grad():
        prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id=None)
        prompts = prompts.to(device)
        tokenized_prompts = tokenized_prompts.to(device)
        compound_prompts_text = [c.to(device) for c in compound_prompts_text]

        text_features = teacher.encode_text_learn(
            prompts, tokenized_prompts, compound_prompts_text).float()
        text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # ========== Student Setup (Trainable) ==========
    logger.info(f"=== Building Student: ViT-B/16@{args.image_size} + DAPM layer {args.student_dpam_layer} ===")
    student_visual = build_student_visual(device, output_dim=768, input_resolution=args.image_size)
    student_visual.DAPM_replace(DPAM_layer=args.student_dpam_layer)

    for param in student_visual.parameters():
        param.requires_grad = True

    # ========== Data ==========
    train_data = Dataset(
        root=args.train_data_path, transform=preprocess,
        target_transform=target_transform, dataset_name=args.dataset, mode='train')

    if args.k_shot > 0:
        indices = get_balanced_few_shot_indices(train_data, args.k_shot)
        bs = min(args.batch_size, len(indices))
        train_dataloader = torch.utils.data.DataLoader(
            train_data, batch_size=bs,
            sampler=torch.utils.data.SubsetRandomSampler(indices))
    else:
        train_dataloader = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size, shuffle=True)

    logger.info(f"Training samples: {len(train_dataloader.dataset)}")

    # ========== Optimizer ==========
    optimizer = torch.optim.Adam(
        student_visual.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True, min_lr=1e-6)

    # ========== Losses ==========
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()

    teacher_features_list = args.teacher_features_list
    student_features_list = args.student_features_list

    logger.info(f"Teacher feature layers: {teacher_features_list}")
    logger.info(f"Student feature layers: {student_features_list}")
    logger.info(f"Student DPAM_layer: {args.student_dpam_layer}")
    logger.info(f"Loss weights - cls: {args.lambda_cls}, sim: {args.lambda_sim}, task: {args.lambda_task}")
    logger.info(f"Distill temperature: {args.distill_temperature}")

    best_loss = float('inf')
    patience = 12
    early_stop_counter = 0

    teacher_img_size = 518
    T = args.distill_temperature

    for epoch in range(args.epoch):
        loss_list = []
        cls_loss_list = []
        sim_loss_list = []
        task_loss_list = []

        student_visual.train()

        pbar = tqdm(train_dataloader, desc=f"Distill Epoch {epoch+1}/{args.epoch}")
        for items in pbar:
            image = items['img'].to(device)
            label = items['anomaly']
            gt = items['img_mask'].squeeze().to(device)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0

            # ---- Teacher Forward (frozen, no_grad) ----
            with torch.no_grad():
                t_img_feat, t_patch_feats = teacher.encode_image(
                    image, teacher_features_list, DPAM_layer=20)
                t_img_feat = t_img_feat / t_img_feat.norm(dim=-1, keepdim=True)

                # Collect teacher RAW LOGIT maps (before softmax) for KL distillation
                logit_scale = 1.0 / 0.07
                t_logit_maps = []
                for idx, pf in enumerate(t_patch_feats):
                    if idx >= args.feature_map_layer[0]:
                        pf = pf / pf.norm(dim=-1, keepdim=True)
                        raw_logits = logit_scale * pf @ text_features[0].t()  # [B, N, 2]
                        logit_map = AnomalyCLIP_lib.get_similarity_map(
                            raw_logits[:, 1:, :], teacher_img_size)  # [B, H_t, W_t, 2]
                        t_logit_maps.append(logit_map)

            # ---- Student Forward ----
            s_img_feat, s_patch_feats = student_visual(
                image, student_features_list, DPAM_layer=args.student_dpam_layer)
            s_img_feat = s_img_feat / s_img_feat.norm(dim=-1, keepdim=True)

            # ---- Distillation Loss 1: CLS Feature Cosine Similarity ----
            loss_cls = (1 - F.cosine_similarity(
                s_img_feat, t_img_feat.detach(), dim=-1)).mean()

            # ---- Student: raw logit maps (for KL distill) + sim maps (for task loss) ----
            s_logit_maps = []
            s_sim_maps = []
            for idx, pf in enumerate(s_patch_feats):
                if idx >= args.feature_map_layer[0]:
                    pf = pf / pf.norm(dim=-1, keepdim=True)
                    raw_logits = logit_scale * pf @ text_features[0].t()  # [B, N, 2]
                    # Raw logit maps for KL distillation
                    logit_map = AnomalyCLIP_lib.get_similarity_map(
                        raw_logits[:, 1:, :], args.image_size)  # [B, H_s, W_s, 2]
                    s_logit_maps.append(logit_map)
                    # Softmax sim maps for task loss
                    sim_map = raw_logits.softmax(dim=-1)
                    sim_map = AnomalyCLIP_lib.get_similarity_map(
                        sim_map[:, 1:, :], args.image_size)  # [B, H_s, W_s, 2]
                    s_sim_maps.append(sim_map)

            # ---- Distillation Loss 2: KL Divergence on Raw Logit Maps ----
            loss_sim = 0
            for t_logit, s_logit in zip(t_logit_maps, s_logit_maps):
                # Resize teacher logit map to student spatial size
                t_logit_resized = F.interpolate(
                    t_logit.permute(0, 3, 1, 2).float(),  # [B, 2, H_t, W_t]
                    size=(args.image_size, args.image_size),
                    mode='bilinear', align_corners=False
                ).permute(0, 2, 3, 1)  # [B, H_s, W_s, 2]

                # Single softmax with temperature on RAW logits
                t_soft = (t_logit_resized / T).softmax(dim=-1)
                s_log_soft = (s_logit / T).log_softmax(dim=-1)

                # KL(t || s) = sum(t * (log(t) - log(s)))
                kl = (t_soft * (t_soft.log() - s_log_soft)).sum(dim=-1).mean()
                loss_sim += kl
            loss_sim = loss_sim / len(t_logit_maps)

            # ---- Task Loss: Focal + Dice + CE (auxiliary regularization) ----
            s_sim_4d = [sm.permute(0, 3, 1, 2) for sm in s_sim_maps]  # [B, 2, H, W]
            loss_pixel = 0
            for sm in s_sim_4d:
                loss_pixel += loss_focal(sm, gt)
                loss_pixel += loss_dice(sm[:, 1, :, :], gt)
                loss_pixel += loss_dice(sm[:, 0, :, :], 1 - gt)
            loss_pixel = loss_pixel / len(s_sim_4d)

            # Image-level CE loss on student
            text_probs = s_img_feat.unsqueeze(1) @ text_features.permute(0, 2, 1)
            text_probs = text_probs[:, 0, ...] / 0.07
            image_loss = F.cross_entropy(text_probs.squeeze(), label.long().to(device))

            task_loss = loss_pixel + image_loss

            # ---- Combined Loss ----
            loss = (
                args.lambda_cls * loss_cls +
                args.lambda_sim * loss_sim +
                args.lambda_task * task_loss
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            cls_loss_list.append(loss_cls.item())
            sim_loss_list.append(loss_sim.item())
            task_loss_list.append(task_loss.item())

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'cls': f"{loss_cls.item():.4f}",
                'sim': f"{loss_sim.item():.4f}",
                'task': f"{task_loss.item():.4f}"
            })

        avg_loss = np.mean(loss_list)
        avg_cls = np.mean(cls_loss_list)
        avg_sim = np.mean(sim_loss_list)
        avg_task = np.mean(task_loss_list)

        logger.info(
            f"Epoch [{epoch+1}/{args.epoch}] "
            f"loss={avg_loss:.4f}  cls={avg_cls:.4f}  "
            f"sim={avg_sim:.4f}  task={avg_task:.4f}"
        )

        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            early_stop_counter = 0
            best_path = os.path.join(args.save_path, 'student_best.pth')
            torch.save(student_visual.state_dict(), best_path)
            logger.info(f"Saved best student -> {best_path}")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        if (epoch + 1) % args.save_freq == 0:
            ckp_path = os.path.join(args.save_path, f'student_epoch_{epoch+1}.pth')
            torch.save(student_visual.state_dict(), ckp_path)

    final_path = os.path.join(args.save_path, 'student_final.pth')
    torch.save(student_visual.state_dict(), final_path)
    logger.info(f"Saved final student -> {final_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("AnomalyCLIP_Distillation", add_help=True)

    parser.add_argument("--train_data_path", type=str, default="./SDD")
    parser.add_argument("--save_path", type=str, default="./checkpoint_distill_sdd")
    parser.add_argument("--dataset", type=str, default="SDD")

    parser.add_argument("--teacher_lora_path", type=str, default="./checkpoint_lora_sdd/best_lora")
    parser.add_argument("--checkpoint_path", type=str, default="./checkpoint_lora_sdd/best_model.pth")

    parser.add_argument("--student_dpam_layer", type=int, default=12)
    parser.add_argument("--student_features_list", type=int, nargs="+", default=[3, 6, 9, 12])
    parser.add_argument("--teacher_features_list", type=int, nargs="+", default=[6, 12, 18, 24])

    parser.add_argument("--image_size", type=int, default=336)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--k_shot", type=int, default=16)
    parser.add_argument("--seed", type=int, default=111)

    parser.add_argument("--depth", type=int, default=9)
    parser.add_argument("--n_ctx", type=int, default=12)
    parser.add_argument("--t_n_ctx", type=int, default=4)
    parser.add_argument("--feature_map_layer", type=int, nargs="+", default=[0, 1, 2, 3])

    parser.add_argument("--print_freq", type=int, default=1)
    parser.add_argument("--save_freq", type=int, default=5)

    parser.add_argument("--lambda_cls", type=float, default=5.0,
                        help="Weight for CLS feature cosine distillation loss")
    parser.add_argument("--lambda_sim", type=float, default=30.0,
                        help="Weight for similarity map KL distillation loss")
    parser.add_argument("--lambda_task", type=float, default=1.0,
                        help="Weight for task loss (Focal+Dice+CE) as auxiliary regularization")
    parser.add_argument("--distill_temperature", type=float, default=2.0,
                        help="Temperature for softening similarity maps in KL distillation")

    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)
    setup_seed(args.seed)
    train_distill(args)
