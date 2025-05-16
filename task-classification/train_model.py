import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. 读取特征数据
df = pd.read_csv("task_features.csv")

# 2. 特征列和标签列
X = df.drop(columns=["task_id", "label"])
y = df["label"]

# 3. 编码标签
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 4. 拆分训练/测试集
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 5. 模型训练
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

# 6. 模型评估
y_pred = model.predict(X_test)
print("分类报告：")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
print("混淆矩阵：")
print(confusion_matrix(y_test, y_pred))

# 7. 保存模型与编码器
import joblib
joblib.dump(model, "hardware_classifier.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
