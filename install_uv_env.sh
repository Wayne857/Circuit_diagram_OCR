#!/bin/bash

# 安装uv并创建Python环境脚本
echo "开始设置uv管理的Python环境..."

# 检查是否已安装uv
if ! command -v uv &> /dev/null; then
    echo "正在安装uv..."
    pip install uv
fi

# 检查uv是否安装成功
if command -v uv &> /dev/null; then
    echo "uv已安装，版本: $(uv --version)"
else
    echo "uv安装失败，请手动安装: pip install uv"
    exit 1
fi

# 创建虚拟环境并安装依赖
echo "创建虚拟环境并安装依赖..."
uv venv .venv
source .venv/bin/activate

# 使用requirements.txt安装依赖
if [ -f "requirements.txt" ]; then
    echo "安装requirements.txt中的依赖..."
    uv pip install -r requirements.txt
elif [ -f "pyproject.toml" ]; then
    echo "安装pyproject.toml中的依赖..."
    uv sync
fi

echo "环境安装完成！"
echo "激活虚拟环境: source .venv/bin/activate"
echo "或者运行: uv run python main.py"