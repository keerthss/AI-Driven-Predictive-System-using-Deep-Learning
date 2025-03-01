import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers
import warnings
warnings.filterwarnings("ignore")

# Load Dataset
data = pd.read_csv('datasets/diabetes.csv')

# Split Features & Target
X = data.drop(columns=['Outcome'], axis=1)
Y = data['Outcome']

# Data Preprocessing
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Build MLP Model
model = tf.keras.Sequential()
model.add(layers.InputLayer(input_shape=(X.shape[1],)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the Model
history = model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_split=0.2)

# Evaluate the Model
test_loss, test_acc = model.evaluate(X_test, Y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Save the Model
model.save("models/diabetes_mlp.h5")
print("Model Saved Successfully ✅")
from performance_graphs import plot_performance
plot_performance(history, "MLP Model")

