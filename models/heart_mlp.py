import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
import numpy as np

# Load the dataset
heart_df = pd.read_csv("./heart.csv")
X = heart_df.iloc[:, :-1].values
y = heart_df.iloc[:, -1].values

# Build the MLP model
model = Sequential()
model.add(Dense(16, activation='relu', input_dim=X.shape[1]))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=100, batch_size=32)

# Save the model
model.save("heart_mlp.h5")
print("Heart Disease Model Trained & Saved Successfully ")
from performance_graphs import plot_performance
plot_performance(history, "MLP Model")

