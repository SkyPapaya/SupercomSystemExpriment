import pandas as pd
import numpy as np
import random

# 设置随机种子以复现
random.seed(42)
np.random.seed(42)

def generate_task(task_id, label):
    if label == 'cpu_sensitive':
        float_op_ratio = np.random.uniform(0.7, 1.0)
        io_call_count = np.random.randint(0, 3)
        mem_access_pattern = 1  # 顺序访问
        cpu_usage = np.random.uniform(70, 100)
        gpu_usage = np.random.uniform(0, 30)
        mem_usage = np.random.uniform(40, 70)

    elif label == 'gpu_sensitive':
        float_op_ratio = np.random.uniform(0.6, 0.95)
        io_call_count = np.random.randint(0, 3)
        mem_access_pattern = np.random.randint(0, 2)
        cpu_usage = np.random.uniform(20, 50)
        gpu_usage = np.random.uniform(70, 100)
        mem_usage = np.random.uniform(60, 90)

    elif label == 'io_sensitive':
        float_op_ratio = np.random.uniform(0.0, 0.3)
        io_call_count = np.random.randint(10, 30)
        mem_access_pattern = 0  # 随机访问
        cpu_usage = np.random.uniform(10, 40)
        gpu_usage = np.random.uniform(0, 20)
        mem_usage = np.random.uniform(30, 60)

    return {
        "task_id": f"T{task_id:03d}",
        "float_op_ratio": round(float_op_ratio, 3),
        "io_call_count": io_call_count,
        "mem_access_pattern": mem_access_pattern,
        "cpu_usage": round(cpu_usage, 2),
        "gpu_usage": round(gpu_usage, 2),
        "mem_usage": round(mem_usage, 2),
        "label": label
    }

# 生成100条任务
tasks = []
labels = ['cpu_sensitive', 'gpu_sensitive', 'io_sensitive']
for i in range(100):
    label = random.choice(labels)
    tasks.append(generate_task(i+1, label))

# 保存为CSV
df = pd.DataFrame(tasks)
df.to_csv('task_features.csv', index=False)
print("✅ 已生成 task_features.csv")
