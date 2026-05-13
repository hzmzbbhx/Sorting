@echo off
REM ==========================================
REM SDD LoRA Teacher Training
REM Uses existing train_lora.py with SDD params
REM ==========================================

set TRAIN_PATH=./SDD
set SAVE_PATH=./checkpoint_lora_sdd
set CHECKPOINT_PATH=./checkpoints/9_12_4_multiscale/epoch_15.pth

echo ==========================================
echo Starting SDD Few-shot LoRA Training...
echo Dataset: SDD (electrical commutators)
echo Train data: %TRAIN_PATH%
echo Save path: %SAVE_PATH%
echo Checkpoint: %CHECKPOINT_PATH%
echo ==========================================

python training\train_lora.py ^
    --train_data_path %TRAIN_PATH% ^
    --save_path %SAVE_PATH% ^
    --checkpoint_path %CHECKPOINT_PATH% ^
    --dataset SDD ^
    --k_shot 16 ^
    --lora_r 8 ^
    --lora_alpha 16 ^
    --epoch 30 ^
    --learning_rate 0.001 ^
    --batch_size 8 ^
    --image_size 518 ^
    --features_list 6 12 18 24 ^
    --feature_map_layer 0 1 2 3 ^
    --depth 9 ^
    --n_ctx 12 ^
    --t_n_ctx 4

echo.
echo ==========================================
echo Training complete.
echo Checkpoint: %SAVE_PATH%\best_model.pth
echo LoRA weights: %SAVE_PATH%\best_lora\
echo ==========================================
pause
