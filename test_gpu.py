import torch


def check_pytorch_gpu():
    """检测 PyTorch 环境下的 GPU 状态"""
    # 1. 检测 GPU 是否可用
    gpu_available = torch.cuda.is_available()
    print(f"✅ GPU 可用状态: {gpu_available}")

    if gpu_available:
        # 2. 查看 GPU 数量
        gpu_count = torch.cuda.device_count()
        print(f"🖥️ GPU 数量: {gpu_count}")

        # 3. 查看当前使用的 GPU 索引和名称
        current_gpu = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_gpu)
        print(f"🎯 当前 GPU 索引: {current_gpu}, 型号: {gpu_name}")

        # 4. 查看 GPU 显存信息
        total_memory = torch.cuda.get_device_properties(current_gpu).total_memory / 1024 ** 3
        print(f"💾 GPU 总显存: {total_memory:.2f} GB")

        # 5. 测试 GPU 计算（创建张量并移到 GPU）
        test_tensor = torch.tensor([1, 2, 3]).to("cuda")
        print(f"✅ GPU 计算测试: 张量 {test_tensor} 已移到 GPU")
    else:
        print("❌ 未检测到 GPU，请检查：")
        print("   - CUDA 驱动是否安装")
        print("   - PyTorch 是否为 GPU 版本（不是 CPU 版本）")
        print("   - 显卡是否支持 CUDA（NVIDIA 显卡）")


if __name__ == "__main__":
    print("===== PyTorch GPU 检测 =====")
    check_pytorch_gpu()