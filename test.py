import joblib
import numpy as np

# 以 E 的 scaler 为例
scaler_E = joblib.load("data/standard_scalerE.pkl")
print("scaler_E 类型:", type(scaler_E))
print("scale_:", scaler_E.scale_)
print("min_:", scaler_E.min_)  # 这个属性保存了转换的偏移量
# 反推：X_original = X_scaled / scale_ + min_