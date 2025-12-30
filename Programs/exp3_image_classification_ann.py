import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.datasets import mnist, cifar10
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def run_experiment(dataset_name, dataset_module, epochs=10):
    print(f"\n--- Loading {dataset_name} Datset ---")
    (X_train, y_train), (X_test, y_test) = dataset_module.load_data()
    
    # Normalize
    X_train, X_test = X_train / 255.0, X_test / 255.0
    
    # Check input shape
    input_shape = X_train.shape[1:]
    num_classes = 10
    
    print(f"Input Shape: {input_shape}")
    
    # Build Deep Feed Forward NN
    model = Sequential([
        Input(shape=input_shape),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    print(f"Training on {dataset_name}...")
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=64, validation_data=(X_test, y_test), verbose=1)
    
    return history

print("Experiment 3: Image Classification with Deep Feed Forward NN")

# Run on MNIST
history_mnist = run_experiment("MNIST", mnist, epochs=5)

# Run on CIFAR-10
history_cifar = run_experiment("CIFAR-10", cifar10, epochs=10) # CIFAR is harder, maybe need more epochs, but keeping 10 for speed

# Plotting
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history_mnist.history['accuracy'], label='Train Acc')
plt.plot(history_mnist.history['val_accuracy'], label='Val Acc')
plt.title('MNIST Accuracy (Deep FFNN)')
plt.xlabel('Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_cifar.history['accuracy'], label='Train Acc')
plt.plot(history_cifar.history['val_accuracy'], label='Val Acc')
plt.title('CIFAR-10 Accuracy (Deep FFNN)')
plt.xlabel('Epochs')
plt.legend()

plt.tight_layout()
plt.savefig('exp3_results.png')
print("\nResults saved to exp3_results.png")
