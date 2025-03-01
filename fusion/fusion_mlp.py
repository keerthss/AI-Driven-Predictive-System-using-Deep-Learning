import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model

# Load Datasets
heart_df = pd.read_csv('./datasets/heart.csv')
diabetes_df = pd.read_csv('./datasets/diabetes.csv')
parkinson_df = pd.read_csv('./datasets/parkinson.csv')

# Preprocessing Heart Dataset
X_heart = heart_df.drop('target', axis=1)
y_heart = heart_df['target']
X_heart = StandardScaler().fit_transform(X_heart)
X_heart_train, X_heart_test, y_heart_train, y_heart_test = train_test_split(X_heart, y_heart, test_size=0.2, random_state=42)

# Preprocessing Diabetes Dataset
X_diabetes = diabetes_df.drop('Outcome', axis=1)
y_diabetes = diabetes_df['Outcome']
X_diabetes = StandardScaler().fit_transform(X_diabetes)
X_diabetes_train, X_diabetes_test, y_diabetes_train, y_diabetes_test = train_test_split(X_diabetes, y_diabetes, test_size=0.2, random_state=42)

# Deep Learning MLP Model
def create_model(input_dim):
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=input_dim))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Training Model for Heart Dataset
model_heart = create_model(X_heart_train.shape[1])
model_heart.fit(X_heart_train, y_heart_train, epochs=50, batch_size=16, verbose=1)
model_heart.save("heart_model.h5")

# Training Model for Diabetes Dataset
model_diabetes = create_model(X_diabetes_train.shape[1])
model_diabetes.fit(X_diabetes_train, y_diabetes_train, epochs=50, batch_size=16, verbose=1)
model_diabetes.save("diabetes_model.h5")

# Fusion Model (Combination)
# Ensure the number of features (columns) match by padding with zeros if necessary
max_columns = max(X_heart_train.shape[1], X_diabetes_train.shape[1])

# Padding the training datasets to have the same number of columns
heart_padding = np.zeros((X_heart_train.shape[0], max_columns - X_heart_train.shape[1]))
diabetes_padding = np.zeros((X_diabetes_train.shape[0], max_columns - X_diabetes_train.shape[1]))

X_heart_train = np.concatenate([X_heart_train, heart_padding], axis=1)
X_diabetes_train = np.concatenate([X_diabetes_train, diabetes_padding], axis=1)

# Create the fusion model
fusion_model = create_model(max_columns)

# Now concatenate and train the fusion model
fusion_model.fit(np.concatenate([X_heart_train, X_diabetes_train], axis=0),
                 np.concatenate([y_heart_train, y_diabetes_train], axis=0),
                 epochs=50, batch_size=16, verbose=1)
fusion_model.save("fusion_mlp.h5")  # Save as fusion_mlp.h5

print("Fusion Model Saved Successfully!")
