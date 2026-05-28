import os
import joblib
import numpy as np
import pandas as pd

SCALER_Z = joblib.load("standard_scalerZ.pkl")
SCALER_A = joblib.load('standard_scalerA.pkl')
SCALER_E = joblib.load("standard_scalerE.pkl")
SCALER_Y = joblib.load("yield_scaler.pkl")

df = pd.read_csv('test_set.csv',header=None)
df.columns = ['z', 'a', 'state', 'results','error']

df[['z']] = SCALER_Z.inverse_transform(df[['z']])
df[['a']] = SCALER_A.inverse_transform(df[['a']])
df[['results']] = SCALER_Y.inverse_transform(df[['results']])

df.to_csv('test_set_scaled.csv',index=False,header=False)