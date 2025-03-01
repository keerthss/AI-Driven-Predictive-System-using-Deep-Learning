

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split  # Added this import
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model

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
fusion_model = load_model("fusion_mlp.h5")

# Padding logic to ensure same number of features (columns)
max_columns = max(X_heart.shape[1], X_diabetes.shape[1])

# Padding the training datasets
heart_padding = np.zeros((X_heart.shape[0], max_columns - X_heart.shape[1]))
diabetes_padding = np.zeros((X_diabetes.shape[0], max_columns - X_diabetes.shape[1]))

X_heart = np.concatenate([X_heart, heart_padding], axis=1)
X_diabetes = np.concatenate([X_diabetes, diabetes_padding], axis=1)

# Split Test Data for Heart Disease and Diabetes
X_heart_train, X_heart_test, y_heart_train, y_heart_test = train_test_split(X_heart, y_heart, test_size=0.2, random_state=42)
X_diabetes_train, X_diabetes_test, y_diabetes_train, y_diabetes_test = train_test_split(X_diabetes, y_diabetes, test_size=0.2, random_state=42)

# Evaluation function
def evaluate_model(model, X_test, y_test, dataset_name):
    print(f"Evaluating {dataset_name} Model...")

    # Ensure the features match the model's input shape
    X_test = X_test[:, :model.input_shape[1]]  # Trim Extra Columns if necessary
    y_pred = (model.predict(X_test) > 0.5).astype(int)

    print(f"{dataset_name} Accuracy: {classification_report(y_test, y_pred)}")

# Evaluate Heart Disease Model
evaluate_model(model_heart, X_heart_test, y_heart_test, "Heart Disease")

# Evaluate Diabetes Model
evaluate_model(model_diabetes, X_diabetes_test, y_diabetes_test, "Diabetes")

# Padding the test datasets to have the same number of columns (for Fusion model evaluation)
heart_test_padding = np.zeros((X_heart_test.shape[0], max_columns - X_heart_test.shape[1]))
diabetes_test_padding = np.zeros((X_diabetes_test.shape[0], max_columns - X_diabetes_test.shape[1]))

X_heart_test = np.concatenate([X_heart_test, heart_test_padding], axis=1)
X_diabetes_test = np.concatenate([X_diabetes_test, diabetes_test_padding], axis=1)

# Now concatenate the padded test datasets and evaluate the fusion model
evaluate_model(fusion_model, np.concatenate([X_heart_test, X_diabetes_test], axis=0),
               np.concatenate([y_heart_test, y_diabetes_test], axis=0), "Fusion")
