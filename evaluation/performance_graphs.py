import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_performance(history, model_name):
    # Accuracy Curve
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['accuracy'], color='purple', linewidth=2)
    plt.plot(history.history['val_accuracy'], color='pink', linewidth=2)
    plt.title(f'{model_name} Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'])
    plt.grid(True)
    plt.show()

    # Loss Curve
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], color='red', linewidth=2)
    plt.plot(history.history['val_loss'], color='orange', linewidth=2)
    plt.title(f'{model_name} Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'])
    plt.grid(True)
    plt.show()
