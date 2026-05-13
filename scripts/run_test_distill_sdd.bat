@echo off
REM ==========================================
REM Teacher vs Student Performance Comparison
REM ==========================================

set DATA_PATH=./SDD
set TEACHER_CKPT=./checkpoint_lora_sdd/best_model.pth
set TEACHER_LORA=./checkpoint_lora_sdd/best_lora
set STUDENT_CKPT=./checkpoint_distill_sdd/student_best.pth

echo ==========================================
echo Comparing Teacher (ViT-L) vs Student (ViT-B)
echo ==========================================

echo.
echo ==========================================
echo [1/2] Testing Teacher (ViT-L/14@336px + LoRA)
echo ==========================================
python inference\test_lora.py ^
    --data_path %DATA_PATH% ^
    --save_path ./results_distill_sdd/teacher ^
    --checkpoint_path %TEACHER_CKPT% ^
    --lora_path %TEACHER_LORA% ^
    --dataset SDD ^
    --features_list 6 12 18 24 ^
    --image_size 518 ^
    --depth 9 ^
    --n_ctx 12 ^
    --t_n_ctx 4 ^
    --metrics image-pixel-level

echo.
echo ==========================================
echo [2/2] Testing Student (ViT-B/16@336 + DAPM layer 12, distilled)
echo ==========================================
python inference\test_distill.py ^
    --data_path %DATA_PATH% ^
    --save_path ./results_distill_sdd/student ^
    --checkpoint_path %TEACHER_CKPT% ^
    --student_checkpoint %STUDENT_CKPT% ^
    --dataset SDD ^
    --features_list 3 6 9 12 ^
    --image_size 336 ^
    --depth 9 ^
    --n_ctx 12 ^
    --t_n_ctx 4 ^
    --student_dpam_layer 12 ^
    --metrics image-pixel-level

echo.
echo ==========================================
echo Comparison complete!
echo Teacher results: results_distill_sdd\teacher\log.txt
echo Student results: results_distill_sdd\student\log.txt
echo ==========================================
pause
