# 超算系统模拟实验
## 目录结构

## 任务流
1. 使用带任务硬件偏好标记的文件进行模型训练 .task-classification/train_model.py
2. 使用带任务硬件偏好标记的文件进行模型评估（可选）
3. 对输入的任务进行拆分，并使用关键任务流分析工具进行关键路径标记,如果是代码，可以使用自动化工具，不是代码请使用人工标记
```
人工标记示例
@task("A")
def load_data():
    pass
```
4. 对各子任务偏好的硬件进行识别，并将其保存为特征文件
5. 利用智能化任务调度系系统开始执行计算任务
## task-simulate
task.py：定义任务结构。

node.py：定义计算节点结构。

predictor.py：实现任务运行时间预测。

priority_scheduler.py：基于任务优先级进行调度。

optimizer.py：提供动态调度优化建议。

main.py：主程序，构建任务依赖图，调用预测器和调度器，展示结果。
## task-classification
### 特征文件