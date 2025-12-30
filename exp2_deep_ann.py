import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load MNIST data
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize data
X_train, X_test = X_train / 255.0, X_test / 255.0

print("Experiment 2: Deep Feed Forward ANN (>=4 Hidden Layers) on MNIST")

def build_deep_model():
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'), # Hidden Layer 1
        Dense(128, activation='relu'), # Hidden Layer 2
        Dense(64, activation='relu'),  # Hidden Layer 3
        Dense(64, activation='relu'),  # Hidden Layer 4
        Dense(32, activation='relu'),  # Hidden Layer 5 (Extra)
        Dense(10, activation='softmax') # Output Layer
    ])
    return model

model = build_deep_model()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

print("\nStarting training...")
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"\nTest Accuracy: {test_acc:.4f}")

# Plotting accuracy
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Deep ANN Training History')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('exp2_results.png')
print("Results saved to exp2_results.png")
