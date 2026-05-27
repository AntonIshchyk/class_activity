import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib

df = pd.read_csv('healthcare-dataset-stroke-data.csv')
print(df.head())
