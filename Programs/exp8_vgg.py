import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Input, Dropout
from tensorflow.keras.datasets import cifar10
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("Experiment 8: VGG-16 & VGG-19 Architecture Implementation (CIFAR-10)")

# Load Data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Encode labels
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

def build_vgg(vgg_type='vgg16', input_shape=(32, 32, 3)):
    model = Sequential()
    model.add(Input(shape=input_shape))
    
    # Block 1
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    
    # Block 2
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    
    # Block 3
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    if vgg_type == 'vgg19':
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same')) # VGG19 has 4 Conv
    else: # VGG16
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    
    # Block 4
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    if vgg_type == 'vgg19':
         model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
         model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    else:
         model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    
    # Block 5
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    if vgg_type == 'vgg19':
         model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
         model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    else:
         model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    
    model.add(Flatten())
    model.add(Dense(512, activation='relu')) # Simplified Dense for CIFAR (Original has 4096)
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    
    return model

# Train VGG-16
print("\n--- Training VGG-16 ---")
model16 = build_vgg('vgg16')
model16.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9), 
                loss='categorical_crossentropy', metrics=['accuracy'])
history16 = model16.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test), verbose=1)

# Train VGG-19 (Optional, uncomment to run, but might be slow. Just building and summarising for demo)
print("\n--- Building VGG-19 (Summary only to save time, code is ready) ---")
model19 = build_vgg('vgg19')
model19.summary()
# history19 = model19.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test), verbose=1)

# Plot VGG-16
plt.figure(figsize=(10, 5))
plt.plot(history16.history['accuracy'], label='VGG16 Train Acc')
plt.plot(history16.history['val_accuracy'], label='VGG16 Val Acc')
# if 'history19' in locals():
#     plt.plot(history19.history['val_accuracy'], label='VGG19 Val Acc')
plt.title('VGG-16 Accuracy on CIFAR-10')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('exp8_vgg_results.png')
print("Results saved to exp8_vgg_results.png")
