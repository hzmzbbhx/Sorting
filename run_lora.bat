@echo off
set DATA_PATH=./Pokerback
set CHECKPOINT_PATH=./checkpoints/9_12_4_multiscale/epoch_15.pth
set LORA_SAVE_PATH=./checkpoint_lora_pokerback

echo ==========================================
echo Start Few-shot LoRA Training...
echo ==========================================
python train_lora.py ^
    --train_data_path %DATA_PATH% ^
    --save_path %LORA_SAVE_PATH% ^
    --checkpoint_path %CHECKPOINT_PATH% ^
    --dataset Pokerback ^
    --k_shot 26 ^
    --lora_r 8 ^
    --lora_alpha 16 ^
    --epoch 30 ^
    --learning_rate 0.001

echo.
