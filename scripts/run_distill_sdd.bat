@echo off
REM ==========================================
REM AnomalyCLIP Knowledge Distillation
REM Teacher: ViT-L/14@336px + LoRA (frozen)
REM Student: ViT-B/16@224 + DAPM (trainable)
REM ==========================================

set TRAIN_PATH=./SDD
set TEACHER_LORA_PATH=./checkpoint_lora_sdd/best_lora
set TEACHER_CKPT=./checkpoint_lora_sdd/best_model.pth
set SAVE_PATH=./checkpoint_distill_sdd

echo ==========================================
echo AnomalyCLIP Knowledge Distillation
echo ==========================================
echo Teacher: ViT-L/14@336px (frozen, with LoRA)
echo Student: ViT-B/16@224 (trainable, with DAPM layer=8)
echo v4 improvements:
echo   DAPM: 8 -^> 12 layers (full dual-path, max anomaly sensitivity)
echo   KL Temperature: 4.0 -^> 2.0 (sharper distributions)
echo   lambda_sim: 5.0 -^> 30.0 (dominant distillation signal)
echo ==========================================

python training\distill.py ^
    --train_data_path %TRAIN_PATH% ^
    --save_path %SAVE_PATH% ^
    --dataset SDD ^
    --teacher_lora_path %TEACHER_LORA_PATH% ^
    --checkpoint_path %TEACHER_CKPT% ^
    --image_size 224 ^
    --batch_size 4 ^
    --epoch 30 ^
    --learning_rate 0.0001 ^
    --k_shot 16 ^
    --student_dpam_layer 12 ^
    --student_features_list 3 6 9 12 ^
    --teacher_features_list 6 12 18 24 ^
    --feature_map_layer 0 1 2 3 ^
    --depth 9 ^
    --n_ctx 12 ^
    --t_n_ctx 4 ^
    --lambda_cls 5.0 ^
    --lambda_sim 30.0 ^
    --lambda_task 1.0 ^
    --distill_temperature 2.0

echo.
echo ==========================================
echo Distillation complete.
echo Student model: %SAVE_PATH%\student_best.pth
echo ==========================================
pause
