import random
import os

def get_balanced_few_shot_indices(dataset, k_shot, random_seed=None):
    """
    从数据集中平衡采样 k_shot 个样本。
    策略：
    1. 将数据集分为 Normal 和 Abnormal 两组。
    2. Abnormal 组优先筛选出有 mask_path 的样本。
    3. 从两组中各随机抽取 k_shot // 2 个样本。
    """
    if random_seed is not None:
        random.seed(random_seed)
    
    normal_indices = []
    abnormal_indices_with_mask = []
    abnormal_indices_no_mask = []

    # 遍历数据集元数据
    # dataset.data_all 是一个列表，包含所有样本的字典信息
    for idx, item in enumerate(dataset.data_all):
        is_anomaly = item['anomaly'] == 1
        has_mask = item['mask_path'] is not None and item['mask_path'] != "" and item['mask_path'] != " "

        if not is_anomaly:
            normal_indices.append(idx)
        else:
            if has_mask:
                abnormal_indices_with_mask.append(idx)
            else:
                abnormal_indices_no_mask.append(idx)

    # 确定采样数量
    n_normal_target = k_shot // 2
    n_abnormal_target = k_shot - n_normal_target

    # 1. 采样正常样本
    if len(normal_indices) >= n_normal_target:
        selected_normal = random.sample(normal_indices, n_normal_target)
    else:
        # 如果正常样本不够，就全部取走
        selected_normal = normal_indices

    # 2. 采样异常样本 (优先取带 Mask 的)
    selected_abnormal = []
    if len(abnormal_indices_with_mask) >= n_abnormal_target:
        selected_abnormal = random.sample(abnormal_indices_with_mask, n_abnormal_target)
    else:
        # 带 Mask 的不够，先全取，再从无 Mask 的补
        selected_abnormal.extend(abnormal_indices_with_mask)
        n_needed = n_abnormal_target - len(selected_abnormal)
        if len(abnormal_indices_no_mask) >= n_needed:
            selected_abnormal.extend(random.sample(abnormal_indices_no_mask, n_needed))
        else:
            selected_abnormal.extend(abnormal_indices_no_mask)

    final_indices = selected_normal + selected_abnormal
    random.shuffle(final_indices) # 再次打乱顺序
    
    print(f"[Sampler] Selected {len(final_indices)} samples: {len(selected_normal)} Normal, {len(selected_abnormal)} Abnormal.")
    return final_indices