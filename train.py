import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib

df = pd.read_csv('healthcare-dataset-stroke-data.csv')
print(df.head())

df.dropna(inplace=True)
print(df.info())
print("====================")

df['gender'] = df['gender'].map({'Male': 1, 'Female': 0, 'Other': 0})
print(df['gender'].value_counts())
print("====================")

print(df['smoking_status'].value_counts())
df['smoking_status'] = df['smoking_status'].map(
    {'never smoked': 0, 'formerly smoked': 1, 'smokes': 2, 'Unknown': -1})
print(df['smoking_status'].value_counts())
print("====================")

FEATURES = ["age", "hypertension", "heart_disease",
            "avg_glucose_level", "bmi", "gender", "smoking_status"]

X = df[FEATURES]
y = df["stroke"]
