from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np

# Load Datasets
heart_df = pd.read_csv('./datasets/heart.csv')
diabetes_df = pd.read_csv('./datasets/diabetes.csv')

# Preprocessing Heart Dataset
X_heart = heart_df.drop('target', axis=1)
y_heart = heart_df['target']

# Preprocessing Diabetes Dataset
X_diabetes = diabetes_df.drop('Outcome', axis=1)
y_diabetes = diabetes_df['Outcome']

# Load Trained Models
model_heart = load_model("heart_model.h5")
model_diabetes = load_model("diabetes_model.h5")

# Predict Classes for both models
y_heart_pred = (model_heart.predict(X_heart) > 0.5).astype(int)
y_diabetes_pred = (model_diabetes.predict(X_diabetes) > 0.5).astype(int)

# Calculate Precision, Recall, F1-Score for Heart Disease Model
precision_heart = precision_score(y_heart, y_heart_pred)
recall_heart = recall_score(y_heart, y_heart_pred)
f1_heart = f1_score(y_heart, y_heart_pred)
roc_auc_heart = roc_auc_score(y_heart, y_heart_pred)

# Calculate Precision, Recall, F1-Score for Diabetes Model
precision_diabetes = precision_score(y_diabetes, y_diabetes_pred)
recall_diabetes = recall_score(y_diabetes, y_diabetes_pred)
f1_diabetes = f1_score(y_diabetes, y_diabetes_pred)
roc_auc_diabetes = roc_auc_score(y_diabetes, y_diabetes_pred)

# Print the classification report for both models
print("Heart Disease Model Classification Report:")
print(classification_report(y_heart, y_heart_pred))

print("\nDiabetes Model Classification Report:")
print(classification_report(y_diabetes, y_diabetes_pred))

# Print individual metrics for both models
print("\nHeart Disease Model Metrics:")
print(f"Precision: {precision_heart:.4f}")
print(f"Recall: {recall_heart:.4f}")
print(f"F1-Score: {f1_heart:.4f}")
print(f"ROC AUC: {roc_auc_heart:.4f}")

print("\nDiabetes Model Metrics:")
print(f"Precision: {precision_diabetes:.4f}")
print(f"Recall: {recall_diabetes:.4f}")
print(f"F1-Score: {f1_diabetes:.4f}")
print(f"ROC AUC: {roc_auc_diabetes:.4f}")
