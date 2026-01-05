import easyocr
import os
import cv2
import numpy as np

# 1. 配置GPU（有则保留，无则注释）
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 2. 初始化EasyOCR（极简配置，避免报错）
reader = easyocr.Reader(
    lang_list=['ch_sim', 'en'],  # 中英双语
    gpu=True,  # 有GPU=True，无则改为False
    verbose=False  # 关闭冗余日志
)

# 3. 【关键】替换为纯英文/数字的图片路径！！！
# 示例：把图片放到桌面，路径改为 r"C:\Users\11\Desktop\test.jpg"
# 绝对不能有中文/特殊字符，否则OpenCV读不到！
IMG_PATH = rf"C:\Users\11\Desktop\pj\image_extract\imgs_test\LDM.png"  # 必须改！改完再运行！

# 4. 强化图片读取容错
def safe_read_img(img_path):
    """安全读取图片，处理路径/编码问题"""
    if not os.path.exists(img_path):
        print(f"❌ 图片不存在：{img_path}")
        return None
    # 解决中文路径读取问题（OpenCV默认不支持中文）
    try:
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print(f"❌ 图片读取失败（可能是格式错误/损坏）：{img_path}")
            return None
        return img
    except Exception as e:
        print(f"❌ 读取图片出错：{e}")
        return None

# 读取图片（容错版）
img = safe_read_img(IMG_PATH)
if img is None:
    exit(1)

# 5. 执行识别（跳过预处理，先用原图测试）
print("🔍 开始识别文本...\n")
try:
    results = reader.readtext(
        img,  # 用读取后的图片对象，而非路径（避免中文路径问题）
        detail=1,
        paragraph=False,
        rotation_info=[0, 90, 180, 270]  # 检测4个方向的文本
    )
except Exception as e:
    print(f"❌ 识别失败：{str(e)}")
    exit(1)

# 6. 终端输出结果
if results and len(results) > 0:
    print(f"✅ 识别到 {len(results)} 行文本：\n")
    for idx, (box, text, score) in enumerate(results):
        # 计算文本倾斜角度（方向）
        (x1, y1), (x2, y2), _, _ = box
        dx = x2 - x1
        dy = y2 - y1
        angle = round(np.degrees(np.arctan2(dy, dx)), 1)

        # 输出
        print(f"--- 文本行 {idx+1} ---")
        print(f"方向：{angle}° | 置信度：{round(score,3)}")
        print(f"内容：{text}\n")
else:
    print("⚠️  未识别到文本（检查图片是否有清晰的文字，或尝试调整rotation_info）")