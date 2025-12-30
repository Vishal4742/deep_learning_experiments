import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Load and preprocess data
def load_data():
    iris = load_iris()
    X = iris.data
    y = iris.target.reshape(-1, 1)
    
    encoder = OneHotEncoder(sparse_output=False)
    y = encoder.fit_transform(y)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_test, y_train, y_test = load_data()

activation_functions = ['sigmoid', 'tanh', 'relu']
histories = {}

print("Experiment 1: Comparing Activation Functions on Iris Dataset")

for activation in activation_functions:
    print(f"\nTraining with {activation} activation...")
    model = Sequential([
        Dense(10, input_shape=(X_train.shape[1],), activation=activation),
        Dense(10, activation=activation),
        Dense(3, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    history = model.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_test, y_test), verbose=0)
    histories[activation] = history
    
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Activation: {activation} - Test Accuracy: {accuracy:.4f}")

# Plotting results
plt.figure(figsize=(10, 6))
for activation in histories:
    plt.plot(histories[activation].history['val_accuracy'], label=f'{activation} val_acc')

plt.title('Validation Accuracy for Different Activation Functions')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('exp1_results.png')
print("\nResults saved to exp1_results.png")
