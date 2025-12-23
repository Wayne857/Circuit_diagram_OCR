import os
import shutil
from pathlib import Path

def organize_files(source_dir, target_base_dir):
    """
    将源文件夹中名称对应的JSON和PNG文件分别整理到目标目录的json和img子文件夹
    
    Args:
        source_dir (str): 源文件夹路径（dlt3）
        target_base_dir (str): 目标基础文件夹路径（segment）
    """
    # 定义目标文件夹路径
    json_target_dir = Path(target_base_dir) / "json"
    img_target_dir = Path(target_base_dir) / "img"
    
    # 创建目标文件夹（如果不存在）
    try:
        json_target_dir.mkdir(parents=True, exist_ok=True)
        img_target_dir.mkdir(parents=True, exist_ok=True)
        print(f"成功创建目标文件夹：\n- JSON文件夹: {json_target_dir}\n- 图片文件夹: {img_target_dir}")
    except Exception as e:
        print(f"创建目标文件夹失败: {e}")
        return
    
    # 校验源文件夹是否存在
    source_path = Path(source_dir)
    if not source_path.exists():
        print(f"错误：源文件夹 {source_dir} 不存在！")
        return
    
    # 获取源文件夹中的所有文件
    all_files = [f for f in source_path.iterdir() if f.is_file()]
    
    # 提取所有文件的基础名称（不含扩展名）并分组
    file_groups = {}
    for file in all_files:
        # 只处理json和png文件
        if file.suffix.lower() in ['.json', '.png']:
            base_name = file.stem  # 获取文件名（不含扩展名）
            if base_name not in file_groups:
                file_groups[base_name] = {'json': None, 'png': None}
            
            if file.suffix.lower() == '.json':
                file_groups[base_name]['json'] = file
            elif file.suffix.lower() == '.png':
                file_groups[base_name]['png'] = file
    
    # 复制匹配的文件到目标文件夹
    copied_count = 0
    skipped_count = 0
    
    for base_name, files in file_groups.items():
        json_file = files['json']
        png_file = files['png']
        
        # 只处理同时存在JSON和PNG的文件对
        if json_file and png_file:
            try:
                # 复制JSON文件
                json_dst = json_target_dir / json_file.name
                shutil.copy2(json_file, json_dst)  # copy2会保留文件元数据
                
                # 复制PNG文件
                png_dst = img_target_dir / png_file.name
                shutil.copy2(png_file, png_dst)
                
                copied_count += 1
                print(f"成功复制: {base_name} (JSON + PNG)")
            except Exception as e:
                print(f"复制文件 {base_name} 失败: {e}")
                skipped_count += 1
        else:
            # 输出缺少对应文件的提示
            missing = []
            if not json_file:
                missing.append("JSON")
            if not png_file:
                missing.append("PNG")
            print(f"跳过 {base_name}: 缺少{', '.join(missing)}文件")
            skipped_count += 1
    
    # 输出统计信息
    print("\n=== 处理完成 ===")
    print(f"成功复制文件对: {copied_count} 组")
    print(f"跳过/失败的文件: {skipped_count} 组")
    print(f"总计处理文件基础名称: {len(file_groups)} 个")

if __name__ == "__main__":
    # 配置路径（可根据实际情况修改）
    SOURCE_DIRECTORY = rf"C:\Users\11\Desktop\images\dlt3"  # 源文件夹路径
    TARGET_BASE_DIRECTORY = rf"./imagessegment"  # 目标基础文件夹
    
    # 执行文件整理
    organize_files(SOURCE_DIRECTORY, TARGET_BASE_DIRECTORY)

    # 执行目标检测的文件整理
    SOURCE_DIRECTORY = rf"C:\Users\11\Desktop\images\text-data"  # 源文件夹路径
    TARGET_BASE_DIRECTORY = rf"./imagesdetect"  # 目标基础文件夹
    organize_files(SOURCE_DIRECTORY, TARGET_BASE_DIRECTORY)
    # yolo train task=segment data=imagessegment\seg\segment.yaml model=ultralytics\weights\yolov8x-seg.pt epochs=200 imgsz=720 batch=8 workers=0 device=0 pretrained=False