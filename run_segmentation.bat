@echo off
chcp 65001 >nul
echo 开始使用train24模型进行分割...

REM 设置Python路径（如果需要）
set PYTHON_CMD=python

REM 使用单张图片进行分割
echo 使用单张图片进行分割示例:
%PYTHON_CMD% segment_single_image.py --image "imagessegment/seg/images/val/Power Monitor_TPS3710DDCR V2.jpg" --model "runs/segment/train24/weights/best.pt" --output "runs/segment/predict_detailed_single" --conf 0.5

echo.
echo 使用文件夹进行分割示例:
%PYTHON_CMD% segment_folder.py --folder "imagessegment/seg/images/val" --model "runs/segment/train24/weights/best.pt" --output "runs/segment/predict_detailed_folder" --conf 0.5

echo.
echo 分割完成！结果已保存到相应目录。
pause