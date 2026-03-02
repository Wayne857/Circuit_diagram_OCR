import os
import shutil
from pathlib import Path

def find_and_copy_target_folders(source_dir, target_root_dir):
    """
    查找包含指定docx文件的文件夹，并复制到目标目录
    
    Args:
        source_dir: 源文件夹路径
        target_root_dir: 目标根目录（dataset文件夹路径）
    """
    # 确保目标根目录存在
    os.makedirs(target_root_dir, exist_ok=True)
    
    # 定义需要查找的关键文件名
    target_docx = "电路描述文档.docx"
    copied_folders = []  # 记录已复制的文件夹
    
    # 遍历源目录的所有层级
    for root, dirs, files in os.walk(source_dir):
        # 检查当前目录是否包含目标docx文件
        if target_docx in files:
            # 获取当前文件夹的名称（最后一级文件夹名）
            folder_name = os.path.basename(root)
            # 目标文件夹路径
            dest_folder = os.path.join(target_root_dir, folder_name)
            
            try:
                # 复制整个文件夹（包括所有文件和子文件）
                # 如果目标文件夹已存在，先删除再复制（避免冲突）
                if os.path.exists(dest_folder):
                    shutil.rmtree(dest_folder)
                shutil.copytree(root, dest_folder)
                copied_folders.append(folder_name)
                print(f"✅ 成功复制文件夹: {folder_name} -> {dest_folder}")
            except Exception as e:
                print(f"❌ 复制文件夹 {folder_name} 失败: {str(e)}")
    
    # 输出总结信息
    print("\n" + "="*50)
    print(f"📊 复制完成！共复制 {len(copied_folders)} 个文件夹到 {target_root_dir}")
    if copied_folders:
        print("📋 复制的文件夹列表:")
        for idx, folder in enumerate(copied_folders, 1):
            print(f"   {idx}. {folder}")
    else:
        print("⚠️  未找到任何包含'电路描述文档.docx'的文件夹")

if __name__ == "__main__":
    # -------------------------- 请修改这里的路径 --------------------------
    # 源文件夹路径（你要遍历的多层级文件夹）
    # 示例：Windows路径 r"C:\Users\你的名字\Desktop\原始数据"
    #       Mac/Linux路径 "/Users/你的名字/Desktop/原始数据"
    SOURCE_FOLDER = rf"experiment/origin"
    
    # ----------------------------------------------------------------------
    
    # 自动计算dataset文件夹路径（源文件夹的同级目录）
    source_parent = os.path.dirname(SOURCE_FOLDER)
    DATASET_FOLDER = os.path.join(source_parent, "dataset")
    
    # 验证源文件夹是否存在
    if not os.path.exists(SOURCE_FOLDER):
        print(f"❌ 错误：源文件夹 {SOURCE_FOLDER} 不存在！")
    else:
        print(f"🔍 开始遍历源文件夹: {SOURCE_FOLDER}")
        find_and_copy_target_folders(SOURCE_FOLDER, DATASET_FOLDER)