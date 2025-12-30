# Deep Learning Lab Programs

This repository contains 10 Python programs based on the Deep Learning Lab Manual.

## Directory Structure

*   `Programs/`: Contains all the Python source code files (`.py`).
*   `Outputs/`: Contains the execution logs (`.txt`) and result plots (`.png`).
*   `run_experiments.ps1`: Script to run all experiments sequentially.

## Experiments Overview

| Exp | Description |
| :--- | :--- |
| 1 | ANN with Backpropagation (varying activation functions) |
| 2 | Deep Feed Forward ANN (>=4 hidden layers) |
| 3 | Image Classification (Deep FFNN) on MNIST/CIFAR-10 |
| 4 | CNN (2 Conv Layers) on MNIST/CIFAR-10 |
| 5 | Data Augmentation Demo |
| 6 | CNN with Data Augmentation |
| 7 | LeNet-5 Architecture |
| 8 | VGG-16 & 19 Architecture |
| 9 | RNN for Sentiment Analysis |
| 10 | Bidirectional LSTM for Sentiment Analysis |

---

## Detailed Results

### EXP1: ANN with Backpropagation
*   **Goal**: Build an ANN, vary activation functions, and compare results.
*   **Code**: [Programs/exp1_ann_backprop.py](Programs/exp1_ann_backprop.py)
*   **Outputs**:
    *   📄 [Execution Log](Outputs/exp1_output.txt)
    *   📊 **Result Plot**:
        
        ![EXP1 Result](Outputs/exp1_results.png)

### EXP2: Deep Feed Forward ANN
*   **Goal**: Build a Deep ANN with >=4 hidden layers.
*   **Code**: [Programs/exp2_deep_ann.py](Programs/exp2_deep_ann.py)
*   **Outputs**:
    *   📄 [Execution Log](Outputs/exp2_output.txt)
    *   📊 **Result Plot**:

        ![EXP2 Result](Outputs/exp2_results.png)

### EXP3: Image Classification (Deep FFNN)
*   **Goal**: Classify MNIST/CIFAR-10 using a Deep Feed Forward NN.
*   **Code**: [Programs/exp3_image_classification_ann.py](Programs/exp3_image_classification_ann.py)
*   **Outputs**:
    *   📄 [Execution Log](Outputs/exp3_output.txt)
    *   📊 **Result Plot**:

        ![EXP3 Result](Outputs/exp3_results.png)

### EXP4: CNN (2 Conv Layers)
*   **Goal**: Classify MNIST/CIFAR-10 using a simple CNN.
*   **Code**: [Programs/exp4_cnn_basic.py](Programs/exp4_cnn_basic.py)
*   **Outputs**:
    *   📄 [Execution Log](Outputs/exp4_output.txt)
    *   📊 **Result Plot**:

        ![EXP4 Result](Outputs/exp4_results.png)

### EXP5: Data Augmentation
*   **Goal**: Demonstrate data augmentation techniques.
*   **Code**: [Programs/exp5_data_augmentation.py](Programs/exp5_data_augmentation.py)
*   **Outputs**:
    *   📄 [Execution Log](Outputs/exp5_output.txt)
    *   🖼️ **Augmentation Demo**:

        ![EXP5 Demo](Outputs/exp5_augmentation_demo.png)

### EXP6: CNN with Data Augmentation
*   **Goal**: Train a CNN on CIFAR-10 with data augmentation.
*   **Code**: [Programs/exp6_cnn_augmentation.py](Programs/exp6_cnn_augmentation.py)
*   **Outputs**:
    *   📄 [Execution Log](Outputs/exp6_output.txt)
    *   📊 **Result Plot**:

        ![EXP6 Result](Outputs/exp6_results.png)

### EXP7: LeNet-5 Architecture
*   **Goal**: Implement LeNet-5 on Fashion MNIST.
*   **Code**: [Programs/exp7_lenet5.py](Programs/exp7_lenet5.py)
*   **Outputs**:
    *   📄 [Execution Log](Outputs/exp7_output.txt)
    *   📊 **Result Plot**:

        ![EXP7 Result](Outputs/exp7_lenet5_results.png)

### EXP8: VGG-16 & 19 Architecture
*   **Goal**: Implement VGG-16/19 on CIFAR-10.
*   **Code**: [Programs/exp8_vgg.py](Programs/exp8_vgg.py)
*   **Outputs**:
    *   📄 [Execution Log](Outputs/exp8_output.txt)
    *   📊 **Result Plot**:

        ![EXP8 Result](Outputs/exp8_vgg_results.png)

### EXP9: RNN for Sentiment Analysis
*   **Goal**: Sentiment analysis on Movie Reviews using SimpleRNN.
*   **Code**: [Programs/exp9_rnn_sentiment.py](Programs/exp9_rnn_sentiment.py)
*   **Outputs**:
    *   📄 [Execution Log](Outputs/exp9_output.txt)
    *   📊 **Result Plot**:

        ![EXP9 Result](Outputs/exp9_rnn_results.png)

### EXP10: Bidirectional LSTM
*   **Goal**: Sentiment analysis using Bidirectional LSTM.
*   **Code**: [Programs/exp10_bi_lstm_sentiment.py](Programs/exp10_bi_lstm_sentiment.py)
*   **Outputs**:
    *   📄 [Execution Log](Outputs/exp10_output.txt)
    *   📊 **Result Plot**:

        ![EXP10 Result](Outputs/exp10_bilstm_results.png)

---

## Operations

**Installation:**
```bash
pip install -r requirements.txt
```

**Run Experiments:**
```powershell
.\run_experiments.ps1
```
