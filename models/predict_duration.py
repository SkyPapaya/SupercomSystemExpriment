import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
import numpy as np

# 读取数据
df = pd.read_csv("../resnet_task_profiles.csv")

# 处理 input_shape 拆解
def parse_input_shape(shape_str):
    try:
        shape = eval(shape_str)
        return pd.Series(shape, index=['input_batch', 'in_ch', 'h', 'w'])
    except:
        return pd.Series([-1, -1, -1, -1], index=['input_batch', 'in_ch', 'h', 'w'])

shape_features = df['input_shape'].apply(parse_input_shape)
df = pd.concat([df, shape_features], axis=1)

# 处理 flops
df['flops_estimate'] = df['flops_estimate'].replace(-1, 0)

# One-hot 编码 task_name
encoder = OneHotEncoder(sparse_output=False)
task_encoded = encoder.fit_transform(df[['task_name']])
task_encoded_df = pd.DataFrame(task_encoded, columns=encoder.get_feature_names_out(['task_name']))

# 合并所有特征
feature_df = pd.concat([
    task_encoded_df,
    df[['gpu_mem_MB', 'gpu_util_percent', 'batch_size', 'input_batch', 'in_ch', 'h', 'w', 'flops_estimate']]
], axis=1)

# 标签：duration_sec
label = df['duration_sec']

# 划分训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(feature_df, label, test_size=0.2, random_state=42)

# 训练 LightGBM 回归模型
model = lgb.LGBMRegressor()
model.fit(X_train, y_train)

# 预测与评估
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print(f"✅ 模型训练完成，测试集 RMSE = {rmse:.4f}")

# 可选：查看特征重要性
import matplotlib.pyplot as plt
lgb.plot_importance(model, max_num_features=10)
plt.tight_layout()
plt.show()
