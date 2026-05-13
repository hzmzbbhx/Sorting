"""
Test distilled student model (ViT-B/16@224 + DAPM).
Reuses the evaluation pipeline from test_lora.py exactly.
"""

import torch
import torch.nn.functional as F
import argparse
import os
import sys
import numpy as np
import random
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import AnomalyCLIP_lib
from AnomalyCLIP_lib.AnomalyCLIP import VisionTransformer
from prompt_ensemble import AnomalyCLIP_PromptLearner
from dataset import Dataset
from logger import get_logger
from utils import get_transform
from metrics import image_level_metrics, pixel_level_metrics
from tabulate import tabulate
from visualization import visualizer


VIT_B_16_URL = "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt"


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def download_vit_b16(cache_dir="./pretrained"):
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, "ViT-B-16.pt")
    if not os.path.exists(fname):
        print(f"[Download] Fetching ViT-B/16 from OpenAI CDN...")
        torch.hub.download_url_to_file(VIT_B_16_URL, fname, progress=True)
    return torch.jit.load(fname, map_location="cpu")


def build_student_visual(device, output_dim=768, input_resolution=224):
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

        for key in student_sd:
            if key in pretrained_sd and student_sd[key].shape == pretrained_sd[key].shape:
                student_sd[key] = pretrained_sd[key].clone()

        if 'proj' in pretrained_sd and 'proj' in student_sd:
            orig = pretrained_sd['proj']
            copy_cols = min(orig.shape[1], student_sd['proj'].shape[1])
            student_sd['proj'][:, :copy_cols] = orig[:, :copy_cols].clone()

        student.load_state_dict(student_sd)
        del jit_model

        # Resize positional embedding if input_resolution differs from pretrained 224
        if input_resolution != 224:
            old_grid = 14
            new_grid = input_resolution // 16
            cls_token = student.positional_embedding.data[:1, :]
            patch_pos = student.positional_embedding.data[1:, :]
            patch_pos = patch_pos.reshape(old_grid, old_grid, 768).permute(2, 0, 1).unsqueeze(0)
            new_patch_pos = F.interpolate(patch_pos, (new_grid, new_grid), mode='bilinear', align_corners=False)
            new_patch_pos = new_patch_pos.squeeze(0).permute(1, 2, 0).reshape(-1, 768)
            import torch.nn as nn
            student.positional_embedding = nn.Parameter(torch.cat([cls_token, new_patch_pos], dim=0))
            print(f"[Student] Resized positional embedding: [{old_grid}x{old_grid}] -> [{new_grid}x{new_grid}]")

    except Exception as e:
        print(f"[Student] Pretrained weight init failed: {e}")

    student.to(device)
    return student


def test_distill(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = get_logger(args.save_path)

    AnomalyCLIP_parameters = {
        "Prompt_length": args.n_ctx,
        "learnabel_text_embedding_depth": args.depth,
        "learnabel_text_embedding_length": args.t_n_ctx
    }

    # Load teacher for text encoder + PromptLearner (no LoRA needed)
    model, _ = AnomalyCLIP_lib.load(
        "ViT-L/14@336px", device=device, design_details=AnomalyCLIP_parameters)
    model.to("cpu")
    # DAPM on teacher is only needed for text encoding (design_details path),
    # not for visual encoding here. But we still need it for encode_text_learn.
    model.visual.DAPM_replace(DPAM_layer=20)
    model.eval()
    model.to(device)

    # Load PromptLearner
    prompt_learner = AnomalyCLIP_PromptLearner(model, AnomalyCLIP_parameters)
    if os.path.exists(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        if "prompt_learner" in checkpoint:
            prompt_learner.load_state_dict(checkpoint["prompt_learner"])
        else:
            prompt_learner.load_state_dict(checkpoint)
        logger.info(f"Loaded PromptLearner from {args.checkpoint_path}")
    prompt_learner.to(device)
    prompt_learner.eval()

    # ---- Student Visual Encoder ----
    logger.info("=== Loading Student: ViT-B/16@224 + DAPM ===")
    student_visual = build_student_visual(device, output_dim=768, input_resolution=args.image_size)
    student_visual.DAPM_replace(DPAM_layer=args.student_dpam_layer)

    if os.path.exists(args.student_checkpoint):
        student_sd = torch.load(args.student_checkpoint, map_location=device)
        student_visual.load_state_dict(student_sd)
        logger.info(f"Loaded student weights from {args.student_checkpoint}")
    else:
        logger.warning(f"Student checkpoint not found: {args.student_checkpoint}")
    student_visual.to(device)
    student_visual.eval()

    # ---- Precompute Text Features ----
    with torch.no_grad():
        prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id=None)
        prompts = prompts.to(device)
        tokenized_prompts = tokenized_prompts.to(device)
        compound_prompts_text = [c.to(device) for c in compound_prompts_text]

        text_features = model.encode_text_learn(
            prompts, tokenized_prompts, compound_prompts_text).float()
        text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # ---- Data ----
    preprocess, target_transform = get_transform(args)
    test_data = Dataset(
        root=args.data_path, transform=preprocess,
        target_transform=target_transform, dataset_name=args.dataset)
    test_dataloader = torch.utils.data.DataLoader(
        test_data, batch_size=1, shuffle=False)
    obj_list = test_data.obj_list

    # ---- Results containers ----
    results = {}
    for obj in obj_list:
        results[obj] = {'gt_sp': [], 'pr_sp': [], 'imgs_masks': [], 'anomaly_maps': []}

    # ---- Inference Loop ----
    logger.info("Starting student inference...")
    for items in tqdm(test_dataloader):
        image = items['img'].to(device)
        cls_name = items['cls_name'][0]

        gt_mask = items['img_mask']
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        results[cls_name]['imgs_masks'].append(gt_mask)
        results[cls_name]['gt_sp'].extend(items['anomaly'].detach().cpu())

        with torch.no_grad():
            image_features, patch_features = student_visual(
                image, args.features_list, DPAM_layer=args.student_dpam_layer)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Image-level score
            text_probs = image_features @ text_features.permute(0, 2, 1)
            text_probs = (text_probs / 0.07).softmax(-1)
            text_probs = text_probs[:, 0, 1]
            results[cls_name]['pr_sp'].extend(text_probs.detach().cpu())

            # Pixel-level anomaly maps
            anomaly_map_list = []
            for idx, patch_feature in enumerate(patch_features):
                if idx >= args.feature_map_layer[0]:
                    patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)
                    similarity, _ = AnomalyCLIP_lib.compute_similarity(
                        patch_feature, text_features[0])
                    similarity_map = AnomalyCLIP_lib.get_similarity_map(
                        similarity[:, 1:, :], args.image_size)
                    anomaly_map = (
                        similarity_map[..., 1] + 1 - similarity_map[..., 0]
                    ) / 2.0
                    anomaly_map_list.append(anomaly_map)

            anomaly_map = torch.stack(anomaly_map_list).sum(dim=0)
            anomaly_map = torch.stack([
                torch.from_numpy(gaussian_filter(i, sigma=args.sigma))
                for i in anomaly_map.detach().cpu()
            ], dim=0)
            results[cls_name]['anomaly_maps'].append(anomaly_map)

            visualizer(
                items['img_path'],
                anomaly_map.detach().cpu().numpy(),
                args.image_size, args.save_path, [cls_name])

    # ---- Metrics (identical to test_lora.py) ----
    table_ls = []
    image_auroc_list = []
    image_ap_list = []
    pixel_auroc_list = []
    pixel_aupro_list = []

    for obj in obj_list:
        table = [obj]
        results[obj]['imgs_masks'] = torch.cat(results[obj]['imgs_masks'])
        results[obj]['anomaly_maps'] = torch.cat(
            results[obj]['anomaly_maps']).detach().cpu().numpy()

        if args.metrics in ['image-level', 'image-pixel-level']:
            image_auroc = image_level_metrics(results, obj, "image-auroc")
            image_ap = image_level_metrics(results, obj, "image-ap")
            image_auroc_list.append(image_auroc)
            image_ap_list.append(image_ap)

        if args.metrics in ['pixel-level', 'image-pixel-level']:
            pixel_auroc = pixel_level_metrics(results, obj, "pixel-auroc")
            pixel_aupro = pixel_level_metrics(results, obj, "pixel-aupro")
            pixel_auroc_list.append(pixel_auroc)
            pixel_aupro_list.append(pixel_aupro)

        if args.metrics == 'pixel-level':
            table.extend([f"{pixel_auroc*100:.1f}", f"{pixel_aupro*100:.1f}"])
        elif args.metrics == 'image-level':
            table.extend([f"{image_auroc*100:.1f}", f"{image_ap*100:.1f}"])
        elif args.metrics == 'image-pixel-level':
            table.extend([
                f"{pixel_auroc*100:.1f}", f"{pixel_aupro*100:.1f}",
                f"{image_auroc*100:.1f}", f"{image_ap*100:.1f}"
            ])

        table_ls.append(table)

    headers = ['objects']
    if args.metrics == 'pixel-level':
        headers.extend(['pixel_auroc', 'pixel_aupro'])
        mean_row = ['mean',
            f"{np.mean(pixel_auroc_list)*100:.1f}",
            f"{np.mean(pixel_aupro_list)*100:.1f}"]
    elif args.metrics == 'image-level':
        headers.extend(['image_auroc', 'image_ap'])
        mean_row = ['mean',
            f"{np.mean(image_auroc_list)*100:.1f}",
            f"{np.mean(image_ap_list)*100:.1f}"]
    elif args.metrics == 'image-pixel-level':
        headers.extend(['pixel_auroc', 'pixel_aupro', 'image_auroc', 'image_ap'])
        mean_row = ['mean',
            f"{np.mean(pixel_auroc_list)*100:.1f}",
            f"{np.mean(pixel_aupro_list)*100:.1f}",
            f"{np.mean(image_auroc_list)*100:.1f}",
            f"{np.mean(image_ap_list)*100:.1f}"]

    table_ls.append(mean_row)
    results_str = tabulate(table_ls, headers=headers, tablefmt="pipe")
    logger.info("\n%s", results_str)
    print(results_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("AnomalyCLIP_Student_Test", add_help=True)

    parser.add_argument("--data_path", type=str, default="./SDD")
    parser.add_argument("--save_path", type=str, default="./results_distill_sdd/student")
    parser.add_argument("--dataset", type=str, default="SDD")

    parser.add_argument("--student_checkpoint", type=str,
                        default="./checkpoint_distill_sdd/student_best.pth")
    parser.add_argument("--checkpoint_path", type=str,
                        default="./checkpoint_lora_sdd/best_model.pth",
                        help="Teacher checkpoint for PromptLearner")

    parser.add_argument("--features_list", type=int, nargs="+", default=[3, 6, 9, 12])
    parser.add_argument("--image_size", type=int, default=336)
    parser.add_argument("--student_dpam_layer", type=int, default=12)
    parser.add_argument("--depth", type=int, default=9)
    parser.add_argument("--n_ctx", type=int, default=12)
    parser.add_argument("--t_n_ctx", type=int, default=4)
    parser.add_argument("--feature_map_layer", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--sigma", type=int, default=4)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--metrics", type=str, default='image-pixel-level',
                        choices=['pixel-level', 'image-level', 'image-pixel-level'])

    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)
    setup_seed(args.seed)
    test_distill(args)
