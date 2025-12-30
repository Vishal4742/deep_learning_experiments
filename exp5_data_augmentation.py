import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import cifar10

# Load a single image from CIFAR-10
(X_train, _), (_, _) = cifar10.load_data()
sample_image = X_train[10] # Pick any image

# Reshape to (1, height, width, channels)
x = sample_image.reshape((1,) + sample_image.shape)

# Create ImageDataGenerator
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

print("Experiment 5: Data Augmentation Demonstration")

# Generate and save/plot augmented images
i = 0
plt.figure(figsize=(10, 4))
plt.subplot(1, 6, 1)
plt.title('Original')
plt.imshow(sample_image)
plt.axis('off')

# .flow() generates batches of randomly transformed images
for batch in datagen.flow(x, batch_size=1):
    i += 1
    ax = plt.subplot(1, 6, i + 1)
    ax.set_title(f'Aug {i}')
    img_plot = plt.imshow(array_to_img(batch[0]))
    plt.axis('off')
    
    if i == 5:
        break

plt.tight_layout()
plt.savefig('exp5_augmentation_demo.png')
print("Augmented images saved to exp5_augmentation_demo.png")
