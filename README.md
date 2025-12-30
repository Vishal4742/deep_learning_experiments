# Deep Learning Lab Programs

This repository contains 10 Python programs based on the Deep Learning Lab Manual.

## Programs List

| Experiment | Description | Program File | Output Log | Result Plot |
| :--- | :--- | :--- | :--- | :--- |
| **EXP1** | Build an Artificial Neural Network by implementing the Backpropagation algorithm and test the same using appropriate data sets. Vary the activation functions used and compare the results. | `exp1_ann_backprop.py` | [View Output](exp1_output.txt) | ![Result](exp1_results.png) |
| **EXP2** | Build a Deep Feed Forward ANN by implementing the Backpropagation algorithm and test the same using appropriate data sets. Use the number of hidden layers >=4. | `exp2_deep_ann.py` | [View Output](exp2_output.txt) | ![Result](exp2_results.png) |
| **EXP3** | Design and implement an Image classification model to classify a dataset of images using Deep Feed Forward NN. Record the accuracy corresponding to the number of epochs. Use the MNIST, CIFAR-10 datasets. | `exp3_image_classification_ann.py` | [View Output](exp3_output.txt) | ![Result](exp3_results.png) |
| **EXP4** | Design and implement a CNN model (with 2 layers of convolutions) to classify multi category image datasets. Record the accuracy corresponding to the number of epochs. Use the MNIST, CIFAR-10 datasets. | `exp4_cnn_basic.py` | [View Output](exp4_output.txt) | ![Result](exp4_results.png) |
| **EXP5** | Use the concept of Data Augmentation to increase the data size from a single image. | `exp5_data_augmentation.py` | [View Output](exp5_output.txt) | ![Result](exp5_augmentation_demo.png) |
| **EXP6** | Design and implement a CNN model to classify CIFAR10 image dataset. Use the concept of Data Augmentation while designing the CNN model. Record the accuracy corresponding to the number of epochs. | `exp6_cnn_augmentation.py` | [View Output](exp6_output.txt) | ![Result](exp6_results.png) |
| **EXP7** | Implement the standard LeNet-5 CNN architecture model to classify multi category image dataset (MNIST, Fashion MNIST) and check the accuracy. | `exp7_lenet5.py` | [View Output](exp7_output.txt) | ![Result](exp7_lenet5_results.png) |
| **EXP8** | Implement the standard VGG-16 & 19 CNN architecture model to classify multi category image dataset and check the accuracy. | `exp8_vgg.py` | [View Output](exp8_output.txt) | ![Result](exp8_vgg_results.png) |
| **EXP9** | Implement RNN for sentiment analysis on movie reviews. | `exp9_rnn_sentiment.py` | [View Output](exp9_output.txt) | ![Result](exp9_rnn_results.png) |
| **EXP10** | Implement Bidirectional LSTM for sentiment analysis on movie reviews. | `exp10_bi_lstm_sentiment.py` | [View Output](exp10_output.txt) | ![Result](exp10_bilstm_results.png) |

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. The programs use TensorFlow/Keras and standard datasets (MNIST, CIFAR-10, IMDB) which will be downloaded automatically on first run.

## Usage

Run all experiments sequentially (Windows PowerShell):
```powershell
.\run_experiments.ps1
```

Or run any program individually:

```bash
python exp1_ann_backprop.py
python exp2_deep_ann.py
# ... and so on
```

## Requirements

- Python 3.8+
- TensorFlow
- NumPy
- Matplotlib
- Scikit-learn

## Course Information

- **Institution**: Oriental College of Technology, Bhopal
- **Department**: CSE-AIML
- **Course**: Deep Learning
- **Program**: B.Tech

## Notes

- All programs are based on the lab manual specifications.
- Programs include example outputs and plots are saved as `.png` files.
- Ensure you have a working internet connection for the first run to download datasets.
