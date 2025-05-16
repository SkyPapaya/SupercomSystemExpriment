import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# 加载采样数据
df = pd.read_csv("../resnet_task_profiles.csv")

# 丢弃无效数据（flops=-1）
df = df[df['flops_estimate'] > 0]

# 选择特征和目标
features = ["gpu_mem_MB", "gpu_util_percent", "batch_size", "flops_estimate"]
X = df[features]
y = df["duration_sec"]

# 拆分训练和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 构建 LightGBM 数据集
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test)

# 设置参数
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'verbose': -1
}

# 训练模型
model = lgb.train(params, train_data, valid_sets=[test_data], early_stopping_rounds=10)

# 预测
y_pred = model.predict(X_test)

# 模型评估
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
print(f"✅ 模型训练完成：RMSE = {rmse:.4f}, R² = {r2:.4f}")

# 保存模型
model.save_model("lightgbm_duration_model.txt")
print("✅ 模型已保存为 lightgbm_duration_model.txt")

# 如果你更喜欢保存为 .pkl 格式（使用 joblib），也可以这样：
joblib.dump(model, "lightgbm_duration_model.pkl")
print("✅ 模型也已保存为 lightgbm_duration_model.pkl")
