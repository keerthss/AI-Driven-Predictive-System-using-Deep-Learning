import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# Load the dataset
parkinson_df = pd.read_csv("parkinson.csv")
# Splitting features and target
X = parkinson_df.drop(columns=["name", "status"], axis=1)

y = parkinson_df['status']

# Normalize Data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Build MLP Model
model = Sequential()
model.add(Dense(128, input_dim=X.shape[1], activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train Model
model.fit(X, y, epochs=100, batch_size=16, verbose=1)

# Evaluate Model
loss, accuracy = model.evaluate(X, y)
print(f"Parkinson Dataset Accuracy: {accuracy * 100:.2f}%")
model.save("parkinsons_model.h5")
print("Model saved as parkinsons_model.h5")
