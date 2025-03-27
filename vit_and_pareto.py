import torch
import timm
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Resize, ToTensor
import pandas as pd

# 加载数据（确保CIFAR-10已下载）
transform = Compose([Resize((224, 224)), ToTensor()])
dataset = CIFAR10(root="./data", train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 模型列表（仅使用本地可用模型）
models = {
    "ResNet18": "resnet18",
    "MobileNetV2": "mobilenetv2_100",
    "ViT-Tiny": "vit_tiny_patch16_224"
}

results = []
for name, model_name in models.items():
    try:
        model = timm.create_model(model_name, pretrained=False, num_classes=10)
        model = model.cuda().eval()
    
    # 测量推理时间
        inputs = torch.randn(1, 3, 224, 224).cuda()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
    
        start.record()
        with torch.no_grad():
            for _ in range(100):  # 100次推理取平均
             _ = model(inputs)
        end.record()
        torch.cuda.synchronize()
        time_ms = start.elapsed_time(end) / 100  # 单次推理时间(ms)
    
        # 测量显存占用
        mem_mb = torch.cuda.max_memory_allocated() / 1024**2
    
        results.append({"Model": name, "Time(ms)": time_ms, "Memory(MB)": mem_mb})
        torch.cuda.reset_peak_memory_stats()

    except Exception as e:
        print(f"加载模型{name}失败: {str(e)}")
        continue

# 输出结果
print(pd.DataFrame(results))