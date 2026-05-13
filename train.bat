@echo off
setlocal enabledelayedexpansion

REM 配置参数
set device=0

REM VisA 数据集训练
set depth=9
set n_ctx=12
set t_n_ctx=4

set base_dir=%depth%_%n_ctx%_%t_n_ctx%_multiscale
set save_dir=checkpoints\%base_dir%\
set LOG=%save_dir%res.log
echo %LOG%

python train.py --dataset visa --train_data_path /remote-home/iot_zhouqihang/data/Visa ^
--save_path %save_dir% ^
--features_list 24 --image_size 518 --batch_size 8 --print_freq 1 ^
--epoch 15 --save_freq 1 --depth %depth% --n_ctx %n_ctx% --t_n_ctx %t_n_ctx%

REM DTD 数据集训练
set base_dir=%depth%_%n_ctx%_%t_n_ctx%_multiscale_DTD
set save_dir=checkpoints\%base_dir%\
set LOG=%save_dir%res.log
echo %LOG%

python train.py --dataset DTD --train_data_path "D:\AnomalyCLIP\DTD-Synthetic" ^
--save_path %save_dir% ^
--features_list 24 --image_size 518 --batch_size 8 --print_freq 1 ^
--epoch 15 --save_freq 1 --depth %depth% --n_ctx %n_ctx% --t_n_ctx %t_n_ctx%

pause