# gpu_utils.py
import torch

def get_gpu_info():
    """封装 GPU 检测逻辑，返回字典格式的 GPU 信息"""
    gpu_info = {"available": False, "count": 0, "name": "", "memory": 0.0}
    if torch.cuda.is_available():
        gpu_info["available"] = True
        gpu_info["count"] = torch.cuda.device_count()
        gpu_info["name"] = torch.cuda.get_device_name(0)
        gpu_info["memory"] = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return gpu_info

if __name__ == "__main__":
    info = get_gpu_info()
    print(f"GPU 可用: {info['available']}")
    print(f"GPU 型号: {info['name']}")