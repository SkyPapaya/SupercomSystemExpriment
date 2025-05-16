import pandas as pd
import joblib

# 1. 加载模型和编码器
model = joblib.load("hardware_classifier.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# 2. 加载待预测特征（和训练结构一致）
task = pd.read_csv("new_task_features.csv")  # 例如从代码分析新生成的一组任务特征
task_id = task["task_id"]
X = task.drop(columns=["task_id", "label"], errors="ignore")

# 3. 预测
y_pred = model.predict(X)
predicted_labels = label_encoder.inverse_transform(y_pred)

# 4. 输出结果
for tid, label in zip(task_id, predicted_labels):
    print(f"任务 {tid} 的推荐执行硬件：{label}")
