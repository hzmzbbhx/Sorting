import torch

# 替换为您实际的 pth 文件路径
ckpt_path = "D:\\test\\AnomalyCLIP\\checkpoints\\9_12_4_multiscale\\epoch_15.pth" 

try:
    # 加载 checkpoint
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    
    print(f"Keys in checkpoint: {checkpoint.keys()}")
    
    if "prompt_learner" in checkpoint:
        state_dict = checkpoint["prompt_learner"]
        print("\n--- Parameters in prompt_learner ---")
        for key, value in state_dict.items():
            print(f"{key:<40} | Shape: {value.shape}")
            
        # 检查是否存在视觉编码器
        has_visual = any("visual" in key for key in state_dict.keys())
        print(f"\nContains Visual Encoder? {'Yes' if has_visual else 'No'}")
    else:
        print("No 'prompt_learner' key found.")

except FileNotFoundError:
    print(f"File not found: {ckpt_path}")
except Exception as e:
    print(f"Error loading checkpoint: {e}")