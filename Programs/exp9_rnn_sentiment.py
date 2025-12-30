import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Embedding, Input
from tensorflow.keras.preprocessing import sequence
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("Experiment 9: RNN for Sentiment Analysis (IMDB)")

# Parameters
max_features = 10000  # Number of words to consider as features
maxlen = 500  # Cut texts after this number of words (among top max_features most common words)
batch_size = 32

# Load Data
print("Loading data...")
(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_features)
print(f"Train sequences: {len(input_train)}")
print(f"Test sequences: {len(input_test)}")

# Pad sequences
print("Pad sequences (samples x time)")
input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
input_test = sequence.pad_sequences(input_test, maxlen=maxlen)
print(f"input_train shape: {input_train.shape}")
print(f"input_test shape: {input_test.shape}")

# Build Model
model = Sequential([
    Input(shape=(maxlen,)),
    Embedding(max_features, 32),
    SimpleRNN(32),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train
print("Training RNN...")
history = model.fit(input_train, y_train,
                    epochs=5, # Reduced epochs for standard RNN as it can be slow/unstable
                    batch_size=128,
                    validation_split=0.2)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('RNN Sentiment Analysis')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('exp9_rnn_results.png')
print("Results saved to exp9_rnn_results.png")
