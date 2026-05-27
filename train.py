import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(
    n_estimators=200, max_depth=8, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred, target_names=['No Stroke', 'Stroke']))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))

joblib.dump( {'model': model, 'features': FEATURES}, filename="stroke_model.pkl")