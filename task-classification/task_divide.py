import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import time
import pynvml
import csv
import os
from torch.fx import symbolic_trace

# 初始化 GPU 监控
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

CSV_PATH = "resnet_fx_task_profiles.csv"

# 初始化 CSV 文件
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "node_name", "op", "target", "duration_sec", "gpu_mem_MB_before", "gpu_mem_MB_after", "gpu_mem_diff_MB"
        ])


def monitor_gpu():
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return mem_info.used / 1024 ** 2  # MB


# 自定义执行模块，逐节点执行并采样
class ProfilerModule(torch.nn.Module):
    def __init__(self, gm):
        super().__init__()
        self.gm = gm
        self.node_results = []

    def forward(self, x):
        env = {}
        for node in self.gm.graph.nodes:
            if node.op == 'placeholder':
                env[node.name] = x
            elif node.op == 'output':
                output = env[node.args[0].name]
                return output
            else:
                # 取输入参数
                args = []
                for arg in node.args:
                    if isinstance(arg, torch.fx.Node):
                        args.append(env[arg.name])
                    else:
                        args.append(arg)
                kwargs = {}
                for k, v in node.kwargs.items():
                    if isinstance(v, torch.fx.Node):
                        kwargs[k] = env[v.name]
                    else:
                        kwargs[k] = v

                # 监控显存，执行操作，计时
                mem_before = monitor_gpu()
                start = time.time()
                # 执行对应操作
                if node.op == 'call_module':
                    submod = dict(self.gm.named_modules())[node.target]
                    out = submod(*args, **kwargs)
                elif node.op == 'call_function':
                    out = node.target(*args, **kwargs)
                elif node.op == 'call_method':
                    # eg: tensor.view()
                    assert len(args) >= 1
                    out = getattr(args[0], node.target)(*args[1:], **kwargs)
                else:
                    raise RuntimeError(f"不支持的节点操作类型: {node.op}")
                end = time.time()
                mem_after = monitor_gpu()

                env[node.name] = out

                # 保存采样结果
                self.node_results.append({
                    "node_name": node.name,
                    "op": node.op,
                    "target": str(node.target),
                    "duration_sec": end - start,
                    "gpu_mem_MB_before": mem_before,
                    "gpu_mem_MB_after": mem_after,
                    "gpu_mem_diff_MB": mem_after - mem_before
                })
        return output


# 数据准备
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 模型与追踪
model = models.resnet18().cuda().eval()
traced = symbolic_trace(model)

profiler = ProfilerModule(traced).cuda()

# 取一批数据测试
batch = next(iter(train_loader))
images, labels = batch
images = images.cuda()

# 运行采样
with torch.no_grad():
    _ = profiler(images)

# 写入 CSV
with open(CSV_PATH, mode='a', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    for res in profiler.node_results:
        writer.writerow([
            res["node_name"], res["op"], res["target"],
            round(res["duration_sec"], 6),
            round(res["gpu_mem_MB_before"], 2),
            round(res["gpu_mem_MB_after"], 2),
            round(res["gpu_mem_diff_MB"], 2)
        ])

print(f"✅ 任务分割采样完成，结果保存在 {CSV_PATH}")
