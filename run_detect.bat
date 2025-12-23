@echo off
echo 正在处理图像...
python main.py --image "C:\Users\11\Desktop\images\dlt3\Acceleration_FXLS8964AFR3 V6.png" --output "predict_res" --classes 1 --process-type whiten
echo 处理完成！
pause