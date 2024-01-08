from xgboost import XGBClassifier
import numpy as np

from sklearn.preprocessing import LabelEncoder
import joblib


# encoder = LabelEncoder()

# Read data from a.txt
with open('a.txt', 'r') as f:
    # Assuming the data is stored as a comma-separated string in the file
    data = np.loadtxt(f, delimiter=',', dtype=float)


data = data.reshape(1, -1)

loaded_model = XGBClassifier()
loaded_model.load_model('xgboost_model_35_67.json')

# # 使用加载的模型进行预测
y_pred = loaded_model.predict(data)


encoder = joblib.load('label_encoder.joblib')

predictions = encoder.inverse_transform([round(value) for value in y_pred])

# # 打印预测结果
print(predictions)
