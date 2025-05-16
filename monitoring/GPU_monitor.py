import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torch.nn.functional as F
import time
import pynvml
import csv
import os
# 初始化 GPU 监控
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

CSV_PATH = "../resnet_task_profiles.csv"

# 初始化 CSV 文件
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "task_name", "duration_sec", "gpu_mem_MB", "gpu_util_percent", "batch_size",
            "input_shape", "flops_estimate"
        ])

def estimate_flops(model, input_tensor):
    try:
        from fvcore.nn import FlopCountAnalysis
        flops = FlopCountAnalysis(model, input_tensor)
        return flops.total() / 1e9  # GFLOPs
    except:
        return -1  # 未安装 fvcore

def monitor_gpu():
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    return mem_info.used / 1024**2, util.gpu  # MB, %

def record_task(name, func, model=None, input_tensor=None, batch_size=None):
    start = time.time()
    mem_before, _ = monitor_gpu()

    result = func()

    mem_after, gpu_util = monitor_gpu()
    end = time.time()

    duration = end - start
    mem_diff = mem_after - mem_before
    flops = estimate_flops(model, input_tensor) if model and input_tensor is not None else -1

    with open(CSV_PATH, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            name, round(duration, 4), round(mem_diff, 2), gpu_util, batch_size,
            tuple(input_tensor.shape) if input_tensor is not None else None, round(flops, 2)
        ])

    return result

# 数据准备
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 模型与优化器
model = models.resnet18().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.train()
times = 0
for batch in train_loader:
    images, labels = batch
    images, labels = images.cuda(), labels.cuda()

    record_task("forward_pass", lambda: model(images), model, images, images.size(0))
    outputs = model(images)
    record_task("loss_computation", lambda: F.cross_entropy(outputs, labels), batch_size=images.size(0))
    loss = F.cross_entropy(outputs, labels)
    record_task("backward_pass", lambda: loss.backward(), batch_size=images.size(0))
    record_task("optimizer_step", lambda: optimizer.step(), batch_size=images.size(0))
    times += 1

    break
    # optimizer.zero_grad()
    # if(time == 100):
    #     break


print("✅ 采样完成，特征已保存至 resnet_task_profiles.csv")
