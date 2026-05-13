from ultralytics import YOLO
import os

# 关键：禁用Ultralytics的所有下载行为
os.environ['ULTRALYTICS_DISABLE_DOWNLOADS'] = 'True'
# 可选：禁用权重缓存（避免缓存干扰）
os.environ['ULTRALYTICS_CACHE'] = 'False'

def train():
    # 1. 加载预训练模型 (推荐使用 yolov8n，速度快且足够检测扑克牌)
    model = YOLO(r"D:\test\AnomalyCLIP\yolov8n.pt")  
    
    # 2. 开始训练
    # data: 指向刚才创建的 yaml 文件
    # epochs: 训练轮数，建议 50-100
    # imgsz: 输入图片大小，640 是标准
    results = model.train(
        data='yolo_poker.yaml', 
        epochs=50, 
        imgsz=640,               # 手机图适配，不用改
        batch=8,                 # 显卡显存小就设4，大就设16
        conf=0.5,                # 置信度阈值，过滤低精度检测
        iou=0.5,                # 多张牌重叠时，调整IOU避免漏检
        device=0,                # 用GPU训练（没有就删，用CPU）
        augment=True             # 数据增强，提升模型泛化能力（必开）
    )
    
    # 3. 验证一下
    metrics = model.val()
    print(f"mAP50: {metrics.box.map50}")

    # 4. 导出模型 (可选，默认会保存在 /poker_detector/weights/best.pt)
    success = model.export(format='onnx')

if __name__ == '__main__':
    train()