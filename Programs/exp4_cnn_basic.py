import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Input
from tensorflow.keras.datasets import mnist, cifar10
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def run_cnn_experiment(dataset_name, dataset_module, epochs=10):
    print(f"\n--- Loading {dataset_name} Datset (CNN) ---")
    (X_train, y_train), (X_test, y_test) = dataset_module.load_data()
    
    # Normalize
    X_train, X_test = X_train / 255.0, X_test / 255.0
    
    # Reshape for MNIST if needed (add channel dim)
    if dataset_name == "MNIST":
        X_train = X_train.reshape((-1, 28, 28, 1))
        X_test = X_test.reshape((-1, 28, 28, 1))
    
    input_shape = X_train.shape[1:]
    num_classes = 10
    
    # Build CNN Model (2 layers of convolutions)
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    print(f"Training CNN on {dataset_name}...")
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=64, validation_data=(X_test, y_test), verbose=1)
    
    return history

print("Experiment 4: CNN (2 Conv Layers) on MNIST and CIFAR-10")

# Run on MNIST
history_mnist = run_cnn_experiment("MNIST", mnist, epochs=5)

# Run on CIFAR-10
history_cifar = run_cnn_experiment("CIFAR-10", cifar10, epochs=10)

# Plotting
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history_mnist.history['accuracy'], label='Train Acc')
plt.plot(history_mnist.history['val_accuracy'], label='Val Acc')
plt.title('MNIST Accuracy (CNN)')
plt.xlabel('Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_cifar.history['accuracy'], label='Train Acc')
plt.plot(history_cifar.history['val_accuracy'], label='Val Acc')
plt.title('CIFAR-10 Accuracy (CNN)')
plt.xlabel('Epochs')
plt.legend()

plt.tight_layout()
plt.savefig('exp4_results.png')
print("\nResults saved to exp4_results.png")
