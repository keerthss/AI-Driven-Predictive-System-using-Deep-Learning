import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
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

# Calculate Confusion Matrix for Heart Disease Model
conf_matrix_heart = confusion_matrix(y_heart, y_heart_pred)

# Calculate Confusion Matrix for Diabetes Model
conf_matrix_diabetes = confusion_matrix(y_diabetes, y_diabetes_pred)

# Plot Confusion Matrix for Heart Disease
plt.figure(figsize=(10, 6))
sns.heatmap(conf_matrix_heart, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix for Heart Disease Model')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Plot Confusion Matrix for Diabetes
plt.figure(figsize=(10, 6))
sns.heatmap(conf_matrix_diabetes, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix for Diabetes Model')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
