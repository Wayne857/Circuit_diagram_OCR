from ultralytics import YOLO

# 加载训练好的模型
model = YOLO(rf"C:\Users\11\Desktop\pj\image_extract\runs\detect\train\weights\best.pt")  # 替换为实际的权重路径

# 执行推理
print(model.predict(
    task="predict",                # 指定任务类型为预测
    source=rf"C:\Users\11\Desktop\images\dlt3\Power Monitor_MAX6421XS29+T V2.png",    # 输入源：图片、视频、文件夹或摄像头（如 0 表示摄像头）
    conf=0.25,                     # 置信度阈值，过滤低置信度目标
    iou=0.45,                      # IOU 阈值，控制目标框的重叠过滤
    save=True,                     # 是否保存预测结果，默认保存到指定目录
    save_txt=False,                # 是否保存预测结果为文本
    save_conf=False,               # 是否保存预测框的置信度值
    show=True,                     # 是否实时显示预测结果
    device=0                       # 使用设备（0 表示第 0 块 GPU，或者 'cpu' 表示使用 CPU）
))
