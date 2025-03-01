import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
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

# Predict probabilities for both models
y_heart_pred_prob = model_heart.predict(X_heart)
y_diabetes_pred_prob = model_diabetes.predict(X_diabetes)

# Calculate ROC curve and AUC for Heart Disease Model
fpr_heart, tpr_heart, _ = roc_curve(y_heart, y_heart_pred_prob)
roc_auc_heart = auc(fpr_heart, tpr_heart)

# Calculate ROC curve and AUC for Diabetes Model
fpr_diabetes, tpr_diabetes, _ = roc_curve(y_diabetes, y_diabetes_pred_prob)
roc_auc_diabetes = auc(fpr_diabetes, tpr_diabetes)

# Plotting ROC Curves
plt.figure(figsize=(10, 6))

# Heart Disease ROC Curve
plt.plot(fpr_heart, tpr_heart, color='blue', lw=2, label=f'Heart Disease (AUC = {roc_auc_heart:.2f})')

# Diabetes ROC Curve
plt.plot(fpr_diabetes, tpr_diabetes, color='green', lw=2, label=f'Diabetes (AUC = {roc_auc_diabetes:.2f})')

# Plot diagonal line (no discrimination)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')

# Labels and Title
plt.title('ROC Curve for Heart Disease and Diabetes Models')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

# Legend
plt.legend(loc='lower right')

# Show the plot
plt.show()
