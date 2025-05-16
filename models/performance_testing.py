import lightgbm as lgb
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import joblib  # 用于模型保存和加载

# 假设你已经有测试集特征和标签
# 这里演示随机数据，实际使用你的X_test和y_test
# X_test = ... (numpy array or pandas dataframe)
# y_test = ...

# 如果你训练后保存了模型，比如用 joblib.dump(model, 'model.pkl')
# 这里加载模型：
model = joblib.load('trained_lightgbm_model.pkl')

# 使用模型进行预测
y_pred = model.predict(X_test)

# 计算 RMSE (均方根误差)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"模型在测试集上的RMSE: {rmse:.4f}")

# 画图对比预测和真实值
plt.figure(figsize=(8,6))
plt.plot(y_test, label='真实值', marker='o')
plt.plot(y_pred, label='预测值', marker='x')
plt.title('真实值 vs 预测值')
plt.xlabel('样本序号')
plt.ylabel('任务耗时/目标值')
plt.legend()
plt.grid(True)
plt.show()
