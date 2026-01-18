# PCB图像文本与元件提取项目

## 环境管理

本项目使用 `uv` 管理Python环境，以确保跨平台的一致性。

### 安装依赖

1. 安装 uv（如果尚未安装）：
   ```bash
   pip install uv
   ```

2. 创建虚拟环境并安装依赖：
   ```bash
   # 方法一：使用脚本安装
   chmod +x install_uv_env.sh
   ./install_uv_env.sh

   # 方法二：手动安装
   uv venv .venv          # 创建虚拟环境
   source .venv/bin/activate  # 激活环境
   uv pip install -r requirements.txt  # 安装依赖
   ```

3. 或者直接使用 uv 同步项目依赖：
   ```bash
   uv sync
   ```

### 运行项目

激活环境后，您可以运行项目：

```bash
# 激活环境
source .venv/bin/activate

# 或者直接运行
uv run python main.py
```

### 环境说明

- Python >= 3.8
- PyTorch (GPU版本)
- Ultralytics YOLO
- OpenCV
- 其他相关依赖

### 依赖锁定

- `requirements.txt` - 精确的包版本列表
- `pyproject.toml` - 项目元数据和依赖声明
- `uv.lock` - uv锁定的依赖版本

### 在新设备上设置环境

只需克隆项目并运行安装脚本：

```bash
git clone <repository-url>
cd image_pcb
chmod +x install_uv_env.sh
./install_uv_env.sh
```

这将自动安装uv、创建虚拟环境并安装所有必要的依赖。