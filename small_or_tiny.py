# small_or_tiny.py（最终稳定版）
import torch
import timm
import matplotlib.pyplot as plt
from torch.optim import Adam
from torchsummary import summary
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Resize, ToTensor
from tqdm import tqdm

def evaluate_model(model, img_size=224):
    """评估模型在测试集上的准确率"""
    transform = Compose([Resize((img_size, img_size)), ToTensor()])
    testset = CIFAR10(root="./data", train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=32, shuffle=False)
    
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def train_small_model(model_name, img_size=224):
    try:
        # 创建模型（强制指定输入尺寸）
        model = timm.create_model(
            model_name,
            num_classes=10,
            pretrained=False,
            img_size=(img_size, img_size),  # 关键修正
            dynamic_img_size=True,
            drop_rate=0.1,
            drop_path_rate=0.1
        )
        model = model.cuda()
        
        # 优化器
        optimizer = Adam(model.parameters(), lr=1e-3)
        
        # 数据加载（优化配置）
        transform = Compose([Resize((img_size, img_size)), ToTensor()])
        dataset = CIFAR10(root="./data", train=True, download=True, transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True
        )
        
        # 训练循环
        scaler = torch.cuda.amp.GradScaler()#混合精度训练+梯度缩放
        model.train()
        pbar = tqdm(dataloader, desc=f"Training {model_name}")
        for inputs, labels in pbar:
            inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = torch.nn.functional.cross_entropy(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # === 必须返回结果 ===
        accuracy = evaluate_model(model, img_size)
        return model, accuracy
        
    except Exception as e:
        print(f"\n[错误] 模型 {model_name} 训练失败：{str(e)}")
        return None, 0.0
def plot_results(results):
    """可视化对比结果"""
    fig, ax1 = plt.subplots(figsize=(10,6))
    
    # 准确率柱状图
    ax1.bar([d['name'] for d in results], [d['acc'] for d in results], 
            color='skyblue', alpha=0.6, label="Accuracy")
    ax1.set_ylabel("Test Accuracy", color='skyblue')
    ax1.tick_params(axis='y', labelcolor='skyblue')
    ax1.set_ylim(0, 1.0)
    
    # 参数量折线图
    ax2 = ax1.twinx()
    ax2.plot([d['name'] for d in results], [d['params'] for d in results], 
            color='red', marker='o', linewidth=2, label="Params (M)")
    ax2.set_ylabel("Parameters (Million)", color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    plt.title("Model Size vs Resolution Impact")
    fig.tight_layout()
    plt.savefig('result.png')
    plt.show()

if __name__ == "__main__":
    # 实验配置（使用实际存在的模型）
    experiments = [
        {"name": "ViT-Tiny (224)", "model": "vit_tiny_patch16_224", "img_size": 224},
        {"name": "Swin-Tiny (224)", "model": "swin_tiny_patch4_window7_224", "img_size": 224},
        ##{"name": "XCiT-Tiny (224)", "model": "xcit_tiny_12_p8_224", "img_size": 224},
        ##{"name": "CvT-13 (224)", "model": "cvt_13", "img_size": 224},
        {"name": "DeiT-Small (224)", "model": "deit_small_patch16_224", "img_size": 224},
        {"name": "EfficientFormerV2-S0 (224)", "model": "efficientformer_l1", "img_size": 224},
        {"name": "ViT-Tiny (384)", "model": "vit_tiny_patch16_224", "img_size": 384},
        {"name": "ViT-Small (384)", "model": "vit_small_patch16_224", "img_size": 384}
    ]
    
    results = []
    for exp in experiments:
        # 训练并评估
        model, acc = train_small_model(exp["model"], exp["img_size"])
        
        # 统计参数量
        params = sum(p.numel() for p in model.parameters()) / 1e6
        
        # 打印模型结构
        print(f"\n=== {exp['name']} ===")
        summary(model, (3, exp["img_size"], exp["img_size"]))
        
        results.append({
            "name": exp["name"],
            "acc": acc,
            "params": params
        })
    
    # 可视化结果
    plot_results(results)
    print("\n实验结果已保存到result.png")