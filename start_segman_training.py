import os
import subprocess
import sys

def setup_environment():
    """设置训练环境"""
    print("正在设置训练环境...")
    
    # 进入SegMAN的segmentation目录
    segman_dir = "C:/Users/11/Desktop/pj/image_extract/SegMAN/segmentation"
    os.chdir(segman_dir)
    print(f"已切换到目录: {segman_dir}")
    
    return segman_dir

def verify_dataset():
    """验证数据集是否正确转换"""
    print("\n正在验证数据集...")
    
    dataset_path = "C:/Users/11/Desktop/pj/image_extract/segman_dataset_fixed"
    
    # 检查目录是否存在
    if not os.path.exists(dataset_path):
        print(f"错误: 数据集路径不存在 {dataset_path}")
        return False
    
    # 检查子目录
    img_train_dir = os.path.join(dataset_path, "img_dir", "train")
    ann_train_dir = os.path.join(dataset_path, "ann_dir", "train")
    img_val_dir = os.path.join(dataset_path, "img_dir", "val")
    ann_val_dir = os.path.join(dataset_path, "ann_dir", "val")
    
    dirs_to_check = [img_train_dir, ann_train_dir, img_val_dir, ann_val_dir]
    dir_names = ["img_dir/train", "ann_dir/train", "img_dir/val", "ann_dir/val"]
    
    for i, dir_path in enumerate(dirs_to_check):
        if os.path.exists(dir_path):
            file_count = len(os.listdir(dir_path))
            print(f"  {dir_names[i]}: 存在, 文件数: {file_count}")
        else:
            print(f"  {dir_names[i]}: 不存在")
            return False
    
    print("数据集验证通过!")
    return True

def start_training():
    """启动训练"""
    print("\n开始训练SegMAN模型...")
    
    config_path = "local_configs/custom/segman_custom.py"
    work_dir = "outputs/segman_custom"
    
    # 创建工作目录
    os.makedirs(work_dir, exist_ok=True)
    
    # 训练命令
    cmd = [
        sys.executable, "tools/train.py",
        config_path,
        "--work-dir", work_dir
    ]
    
    print(f"运行命令: {' '.join(cmd)}")
    
    try:
        # 启动训练进程
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        print("训练已启动! 实时输出:")
        print("-" * 50)
        
        # 实时输出日志
        for line in iter(process.stdout.readline, ''):
            print(line.rstrip())
        
        # 等待进程结束
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            print(f"\n训练过程中出现错误，返回码: {process.returncode}")
            print(f"错误输出: {stderr}")
            return False
        else:
            print(f"\n训练成功完成!")
            return True
            
    except Exception as e:
        print(f"启动训练时出现异常: {str(e)}")
        return False

def main():
    print("SegMAN模型训练启动器")
    print("="*50)
    
    # 设置环境
    segman_dir = setup_environment()
    
    # 验证数据集
    if not verify_dataset():
        print("\n数据集验证失败，退出。")
        return
    
    # 询问用户是否开始训练
    response = input("\n数据集验证通过，是否开始训练? (y/n): ").lower().strip()
    if response in ['y', 'yes', '是']:
        success = start_training()
        if success:
            print("\n训练完成! 您可以在 outputs/segman_custom 目录下查看结果。")
        else:
            print("\n训练失败，请检查错误信息。")
    else:
        print("训练已取消。")
    
    print("\n训练脚本执行完毕。")

if __name__ == "__main__":
    main()
