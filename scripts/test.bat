@echo off
setlocal enabledelayedexpansion

REM 配置参数
set device=0
set depth=9
set n_ctx=12
set t_n_ctx=4

REM mvtec 测试
set base_dir=%depth%_%n_ctx%_%t_n_ctx%_multiscale
set save_dir=checkpoints\%base_dir%\
python test.py --dataset mvtec ^
    --data_path /remote-home/iot_zhouqihang/data/mvdataset --save_path results\%base_dir%\zero_shot ^
    --checkpoint_path %save_dir%epoch_15.pth ^
    --features_list 24 --image_size 518 --depth %depth% --n_ctx %n_ctx% --t_n_ctx %t_n_ctx%

REM DTD 测试
set base_dir=%depth%_%n_ctx%_%t_n_ctx%_multiscale
set save_dir=checkpoints\%base_dir%\
python test.py --dataset DTD ^
    --data_path "D:\test\AnomalyCLIP\DTD-Synthetic" ^
    --save_path results\%base_dir%\zero_shot ^
    --checkpoint_path %save_dir%epoch_15.pth ^
    --features_list 24 --image_size 518 --depth %depth% --n_ctx %n_ctx% --t_n_ctx %t_n_ctx%

pause
