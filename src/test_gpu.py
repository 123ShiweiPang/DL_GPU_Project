import os
import torch

# ===================== 路径配置（核心：自动识别项目根目录） =====================
# 获取当前脚本（test_gpu.py）的绝对路径
CURRENT_FILE_PATH = os.path.abspath(__file__)
# 获取src目录路径（当前脚本的父目录）
SRC_DIR = os.path.dirname(CURRENT_FILE_PATH)
# 获取项目根目录（src目录的父目录）
ROOT_DIR = os.path.dirname(SRC_DIR)
# 定义数据/模型目录路径（基于根目录，永不跑偏）
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODELS_DIR = os.path.join(ROOT_DIR, "models")


# ===================== GPU 检测核心功能 =====================
def get_gpu_info():
    """封装GPU检测逻辑，返回结构化的GPU信息"""
    gpu_info = {
        "available": False,
        "count": 0,
        "name": "",
        "total_memory_gb": 0.0,
        "cuda_version": torch.version.cuda if hasattr(torch.version, "cuda") else "N/A"
    }

    if torch.cuda.is_available():
        gpu_info["available"] = True
        gpu_info["count"] = torch.cuda.device_count()
        gpu_info["name"] = torch.cuda.get_device_name(0)
        # 计算总显存（转换为GB）
        gpu_info["total_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / (1024 ** 3),
            2
        )
    return gpu_info


# ===================== 主函数（执行+验证路径） =====================
if __name__ == "__main__":
    # 打印路径信息（验证是否正确）
    print("=" * 50)
    print("📁 项目路径验证：")
    print(f"项目根目录: {ROOT_DIR}")
    print(f"源代码目录: {SRC_DIR}")
    print(f"数据目录: {DATA_DIR}")
    print(f"模型目录: {MODELS_DIR}")
    print("=" * 50)

    # 检测并打印GPU信息
    gpu_info = get_gpu_info()
    print("\n🖥️ GPU 检测结果：")
    print(f"GPU 是否可用: {gpu_info['available']}")
    if gpu_info["available"]:
        print(f"GPU 数量: {gpu_info['count']}")
        print(f"GPU 型号: {gpu_info['name']}")
        print(f"GPU 总显存: {gpu_info['total_memory_gb']} GB")
        print(f"CUDA 版本: {gpu_info['cuda_version']}")
    else:
        print("⚠️ 未检测到可用GPU，将使用CPU运行")
    print("=" * 50)