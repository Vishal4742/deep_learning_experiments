# Deep Learning Experiments

This repository contains implementation of 10 Deep Learning experiments using TensorFlow/Keras.

## Experiment List

| Experiment | Description | Program File | Output Log | Result Plot |
| :--- | :--- | :--- | :--- | :--- |
| **EXP1** | ANN with Backpropagation (varying activation functions) | `exp1_ann_backprop.py` | [View Output](exp1_output.txt) | ![Result](exp1_results.png) |
| **EXP2** | Deep Feed Forward ANN (>=4 hidden layers) | `exp2_deep_ann.py` | [View Output](exp2_output.txt) | ![Result](exp2_results.png) |
| **EXP3** | Image Classification (Deep FFNN) | `exp3_image_classification_ann.py` | [View Output](exp3_output.txt) | ![Result](exp3_results.png) |
| **EXP4** | CNN (2 Conv Layers) | `exp4_cnn_basic.py` | [View Output](exp4_output.txt) | ![Result](exp4_results.png) |
| **EXP5** | Data Augmentation Demo | `exp5_data_augmentation.py` | [View Output](exp5_output.txt) | ![Result](exp5_augmentation_demo.png) |
| **EXP6** | CNN with Data Augmentation | `exp6_cnn_augmentation.py` | [View Output](exp6_output.txt) | ![Result](exp6_results.png) |
| **EXP7** | LeNet-5 Architecture | `exp7_lenet5.py` | [View Output](exp7_output.txt) | ![Result](exp7_lenet5_results.png) |
| **EXP8** | VGG-16 & 19 Architecture | `exp8_vgg.py` | [View Output](exp8_output.txt) | ![Result](exp8_vgg_results.png) |
| **EXP9** | RNN for Sentiment Analysis | `exp9_rnn_sentiment.py` | [View Output](exp9_output.txt) | ![Result](exp9_rnn_results.png) |
| **EXP10** | Bidirectional LSTM for Sentiment Analysis | `exp10_bi_lstm_sentiment.py` | [View Output](exp10_output.txt) | ![Result](exp10_bilstm_results.png) |

## How to Run

Dependencies: `tensorflow`, `numpy`, `matplotlib`, `scikit-learn`.

You can run all experiments sequentially using the provided PowerShell script (Windows):
```powershell
.\run_experiments.ps1
```

Or run individual files:
```bash
python exp1_ann_backprop.py
```
