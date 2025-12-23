import os
from pathlib import Path

class Config:
    """配置管理类"""
    
    def __init__(self):
        """初始化配置"""
        self.project_root = Path(__file__).parent.parent.absolute()
        self.setup_paths()
    
    def setup_paths(self):
        """设置项目路径"""
        # 检测模型路径
        self.detection_model_path = self.project_root / "runs" / "detect" / "train" / "weights" / "best.pt"
        
        # 分割模型路径
        self.segmentation_model_path = self.project_root / "runs" / "segment" / "train24" / "weights" / "best.pt"
        
        # 输入图像路径
        self.input_image_dir = self.project_root / "test_IMG"
        
        # 输出路径
        self.output_dir = self.project_root / "predict_res"
        self.output_dir.mkdir(exist_ok=True)
        
        # 检测结果保存路径
        self.detection_output_dir = self.output_dir / "detection_results"
        self.detection_output_dir.mkdir(exist_ok=True)
        
        # 处理后图像保存路径
        self.processed_output_dir = self.output_dir / "processed_images"
        self.processed_output_dir.mkdir(exist_ok=True)
    
    def get_input_images(self):
        """获取输入图像列表"""
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        images = []
        
        if self.input_image_dir.exists():
            for file in self.input_image_dir.iterdir():
                if file.suffix.lower() in image_extensions:
                    images.append(str(file))
                    
        return images
    
    def get_output_paths(self, image_path):
        """根据输入图像路径生成输出路径"""
        image_name = Path(image_path).stem
        image_ext = Path(image_path).suffix
        
        # 检测结果路径
        detection_result_path = self.detection_output_dir / f"{image_name}_result{image_ext}"
        
        # 处理后图像路径
        processed_image_path = self.processed_output_dir / f"{image_name}_whitened{image_ext}"
        
        return str(detection_result_path), str(processed_image_path)