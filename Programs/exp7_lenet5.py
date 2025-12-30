import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, AveragePooling2D, Input
from tensorflow.keras.datasets import fashion_mnist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("Experiment 7: LeNet-5 Implementation (Fashion MNIST)")

# Load Data
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Preprocess
X_train = X_train.reshape((-1, 28, 28, 1)) / 255.0
X_test = X_test.reshape((-1, 28, 28, 1)) / 255.0

# LeNet-5 Architecture
# Note: Original LeNet-5 used 32x32, we pad 28x28 to 32x32 or just accept 28x28.
# Implementation adapted for 28x28 input.
model = Sequential([
    Input(shape=(28, 28, 1)),
    
    # C1: 6 filters, 5x5 kernel, tanh (traditional) or relu
    Conv2D(6, (5, 5), activation='tanh', padding='same'),
    
    # S2: Avg Pooling 2x2
    AveragePooling2D(pool_size=(2, 2)),
    
    # C3: 16 filters, 5x5 kernel
    Conv2D(16, (5, 5), activation='tanh'),
    
    # S4: Avg Pooling 2x2
    AveragePooling2D(pool_size=(2, 2)),
    
    # C5: 120 filters, 5x5 kernel (becomes fully connected if input size matches)
    Conv2D(120, (5, 5), activation='tanh'),
    
    Flatten(),
    
    # F6: 84 units
    Dense(84, activation='tanh'),
    
    # Output: 10 units (softmax for modern classification)
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

print("Training LeNet-5...")
history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

# Plot
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('LeNet-5 on Fashion MNIST')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('exp7_lenet5_results.png')
print("Results saved to exp7_lenet5_results.png")
