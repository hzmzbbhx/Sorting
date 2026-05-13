import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
from prompt_ensemble import AnomalyCLIP_PromptLearner
from dataset import Dataset
from logger import get_logger
from tqdm import tqdm
import numpy as np
from utils import get_transform
import AnomalyCLIP_lib
from peft import PeftModel
from scipy.ndimage import gaussian_filter
from metrics import image_level_metrics, pixel_level_metrics
from tabulate import tabulate
import random
from visualization import visualizer

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def test_lora(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = get_logger(args.save_path)

    # 1. 加载基础模型配置
    AnomalyCLIP_parameters = {
        "Prompt_length": args.n_ctx, 
        "learnabel_text_embedding_depth": args.depth, 
        "learnabel_text_embedding_length": args.t_n_ctx
    }
    
    # 加载 CLIP 模型
    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details=AnomalyCLIP_parameters)

    # [关键步骤] 先替换 DAPM 结构，确保结构与训练时一致
    model.visual.DAPM_replace(DPAM_layer=20)

    # 2. 加载 LoRA 权重到 model.visual
    # 注意：必须加载到 visual 部分，因为训练时是对 visual 做的 LoRA
    try:
        model.visual = PeftModel.from_pretrained(model.visual, args.lora_path)
        logger.info(f"Successfully loaded LoRA weights from {args.lora_path}")
    except Exception as e:
        logger.error(f"Failed to load LoRA weights: {e}")
        raise e
        
    model.eval()
    model.to(device)

    # 3. 加载 Prompt Learner
    prompt_learner = AnomalyCLIP_PromptLearner(model, AnomalyCLIP_parameters)
    if os.path.exists(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        # 兼容不同的 checkpoint 保存格式
        if "prompt_learner" in checkpoint:
            prompt_learner.load_state_dict(checkpoint["prompt_learner"])
        else:
            prompt_learner.load_state_dict(checkpoint)
        logger.info(f"Loaded prompt learner from {args.checkpoint_path}")
    
    prompt_learner.to(device)
    prompt_learner.eval()

    # 4. 预计算文本特征 (Text Features)
    # 参考 test.py 的逻辑，提前计算好文本特征以加速推理
    with torch.no_grad():
        prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id=None)
        # 确保数据在 device 上
        if isinstance(prompts, torch.Tensor): prompts = prompts.to(device)
        if isinstance(tokenized_prompts, torch.Tensor): tokenized_prompts = tokenized_prompts.to(device)
        if isinstance(compound_prompts_text, torch.Tensor): compound_prompts_text = compound_prompts_text.to(device)
        
        text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
        text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # 5. 数据加载
    preprocess, target_transform = get_transform(args)
    test_data = Dataset(root=args.data_path, transform=preprocess, target_transform=target_transform, dataset_name=args.dataset)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    
    # 初始化结果容器
    obj_list = test_data.obj_list
    results = {}
    for obj in obj_list:
        results[obj] = {'gt_sp': [], 'pr_sp': [], 'imgs_masks': [], 'anomaly_maps': []}

    # 6. 推理循环
    logger.info("Starting inference...")
    for items in tqdm(test_dataloader):
        image = items['img'].to(device)
        cls_name = items['cls_name'][0]
        
        # 处理 Ground Truth Mask
        gt_mask = items['img_mask']
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        results[cls_name]['imgs_masks'].append(gt_mask)
        results[cls_name]['gt_sp'].extend(items['anomaly'].detach().cpu())

        with torch.no_grad():
            # 编码图像 (包含 LoRA 权重)
            image_features, patch_features = model.encode_image(image, args.features_list, DPAM_layer=20)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # 计算图像级得分
            text_probs = image_features @ text_features.permute(0, 2, 1)
            text_probs = (text_probs/0.07).softmax(-1)
            text_probs = text_probs[:, 0, 1]
            results[cls_name]['pr_sp'].extend(text_probs.detach().cpu())

            # 计算像素级异常图
            anomaly_map_list = []
            for idx, patch_feature in enumerate(patch_features):
                if idx >= args.feature_map_layer[0]:
                    patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)
                    similarity, _ = AnomalyCLIP_lib.compute_similarity(patch_feature, text_features[0])
                    similarity_map = AnomalyCLIP_lib.get_similarity_map(similarity[:, 1:, :], args.image_size)
                    anomaly_map = (similarity_map[..., 1] + 1 - similarity_map[..., 0]) / 2.0
                    anomaly_map_list.append(anomaly_map)

            # 融合特征层并应用高斯滤波
            anomaly_map = torch.stack(anomaly_map_list).sum(dim=0)
            anomaly_map = torch.stack([torch.from_numpy(gaussian_filter(i, sigma=args.sigma)) for i in anomaly_map.detach().cpu()], dim=0)
            results[cls_name]['anomaly_maps'].append(anomaly_map)
            
            # 可视化保存
            visualizer(items['img_path'], anomaly_map.detach().cpu().numpy(), args.image_size, args.save_path, [cls_name])

    # 7. 指标计算 (完全参考 test.py)
    table_ls = []
    image_auroc_list = []
    image_ap_list = []
    pixel_auroc_list = []
    pixel_aupro_list = []

    for obj in obj_list:
        table = []
        table.append(obj)
        
        # 拼接结果
        results[obj]['imgs_masks'] = torch.cat(results[obj]['imgs_masks'])
        results[obj]['anomaly_maps'] = torch.cat(results[obj]['anomaly_maps']).detach().cpu().numpy()
        
        # 根据 metrics 参数计算不同指标
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

        # 构建表格行
        if args.metrics == 'pixel-level':
            table.append(f"{pixel_auroc*100:.1f}")
            table.append(f"{pixel_aupro*100:.1f}")
        elif args.metrics == 'image-level':
            table.append(f"{image_auroc*100:.1f}")
            table.append(f"{image_ap*100:.1f}")
        elif args.metrics == 'image-pixel-level':
            table.append(f"{pixel_auroc*100:.1f}")
            table.append(f"{pixel_aupro*100:.1f}")
            table.append(f"{image_auroc*100:.1f}")
            table.append(f"{image_ap*100:.1f}")
            
        table_ls.append(table)

    # 8. 计算均值并输出表格
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
    parser = argparse.ArgumentParser("AnomalyCLIP_LoRA_Test", add_help=True)
    # Data Paths
    parser.add_argument("--data_path", type=str, default="./data/visa", help="path to test dataset")
    parser.add_argument("--save_path", type=str, default='./results_lora/', help='path to save results')
    
    # Weights Paths
    parser.add_argument("--checkpoint_path", type=str, default='./checkpoint/epoch_15.pth', help='path to prompt learner checkpoint')
    parser.add_argument("--lora_path", type=str, default='./checkpoint_lora', help='path to lora weights folder')
    
    # Model Params
    parser.add_argument("--dataset", type=str, default='mvtec')
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24])
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--depth", type=int, default=9)
    parser.add_argument("--n_ctx", type=int, default=12)
    parser.add_argument("--t_n_ctx", type=int, default=4)
    parser.add_argument("--feature_map_layer", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--sigma", type=int, default=4)
    parser.add_argument("--seed", type=int, default=111)
    
    # Metrics Control
    parser.add_argument("--metrics", type=str, default='image-pixel-level', choices=['pixel-level', 'image-level', 'image-pixel-level'])
    
    args = parser.parse_args()
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        
    setup_seed(args.seed)
    test_lora(args)