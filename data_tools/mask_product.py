import json
import numpy as np
import cv2
import os
import glob
from labelme import utils # 需要 pip install labelme

json_dir = r"D:\test\AnomalyCLIP\Pokerback\card\ground_truth\rotten" # 放json的文件夹
save_dir = r"D:\test\AnomalyCLIP\Pokerback\card\ground_truth\rotten" # 保存mask的文件夹

if not os.path.exists(save_dir): os.makedirs(save_dir)

json_files = glob.glob(os.path.join(json_dir, "*.json"))

for json_file in json_files:
    data = json.load(open(json_file))
    
    # 获取图片形状
    img_h = data['imageHeight']
    img_w = data['imageWidth']
    
    # 生成空的掩码
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    
    # 遍历所有标注形状
    for shape in data['shapes']:
        # 获取多边形点
        points = shape['points']
        # 将点转换为轮廓格式
        points = np.array(points, dtype=np.int32)
        # 在掩码上填充白色 (255)
        cv2.fillPoly(mask, [points], 255)
        
    # 保存
    filename = os.path.basename(json_file).replace(".json", ".png")
    cv2.imwrite(os.path.join(save_dir, filename), mask)