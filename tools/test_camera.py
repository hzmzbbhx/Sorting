import cv2

# 修改这里：0 或 1
camera_index = 0

cap = cv2.VideoCapture(camera_index)

if cap.isOpened():
    print("---------------------------------------")
    # 1. 先看默认是多少
    d_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    d_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"默认分辨率: {int(d_w)} x {int(d_h)}")

    # 2. 尝试强制设置为高清 (1280x720)
    print("正在尝试切换到 1280 x 720 ...")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # 3. 再次读取，看是否生效
    ret, frame = cap.read()
    if ret:
        h, w, _ = frame.shape
        print(f"实际生效分辨率: {w} x {h}")
        if w == 1280:
            print("✅ 恭喜！你的摄像头支持高清，Config里请填 1280 和 720")
        else:
            print("❌ 遗憾，摄像头不支持高清，Config里只能填 640 和 480")
    print("---------------------------------------")
else:
    print("无法打开摄像头")

cap.release()