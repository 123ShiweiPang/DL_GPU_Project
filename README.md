# DL_GPU_Project
基于 GPU 加速的深度学习项目模板，专注于高效利用 GPU 资源完成模型训练/推理任务。

## 📋 项目介绍
本项目为深度学习 GPU 加速实践模板，包含基础的 GPU 环境检测、代码加速示例，可快速适配各类深度学习框架（PyTorch/TensorFlow 等）。

## 🛠️ 环境依赖
### 硬件要求
- GPU：NVIDIA RTX 3060/4060/5060 及以上（支持 CUDA）
- 显存：8GB 及以上

### 软件要求
```bash
# 基础依赖（建议用 conda 或 venv 创建虚拟环境）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas matplotlib
pip install nvidia-ml-py3  # 可选：GPU 状态监控