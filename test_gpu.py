import torch

# 核心 GPU 检测逻辑
print("=== GPU 检测结果 ===")
print("PyTorch 版本:", torch.__version__)
print("CUDA 是否可用:", torch.cuda.is_available())

# 如果检测到 GPU，输出详细信息
if torch.cuda.is_available():
    print("GPU 名称:", torch.cuda.get_device_name(0))
    print("GPU 数量:", torch.cuda.device_count())
    # 测试 GPU 张量运算（验证算力）
    gpu_tensor = torch.tensor([1, 2, 3]).cuda()
    print("GPU 张量示例:", gpu_tensor, "（设备：", gpu_tensor.device, "）")
else:
    print("⚠️ 未检测到 GPU，仅使用 CPU")