COVID-19 Detection from Chest X-Ray Images using CNN and Transfer Learning

This repository contains the complete implementation for a research reproduction and extension project based on the paper:
“Artificial Intelligence Applied to Chest X-Ray Images for the Automatic Detection of COVID-19: A Thoughtful Evaluation Approach”
IEEE Xplore link

The project replicates the key methodology of the paper using public datasets and extends it with a three-class classification task — distinguishing between COVID-19, Normal, and Viral Pneumonia chest X-rays.

Project Structure

The repository is organized into two main parts:

1_Baseline_CNN/

Contains the reproduction of the original paper’s baseline experiment using a custom CNN architecture trained on a balanced subset of the COVID-19 Radiography Database from Kaggle.

Kaggle link: https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database
/data 

notebooks/ — Jupyter/Colab notebook with full code.

results/ — Sample output images and confusion matrix visualizations.

2_TransferLearning_Extension/

Extends the baseline to include:

A stronger pretrained backbone (MobileNetV2) for the same binary task.

A 3-class extension distinguishing COVID-19, Normal, and Viral Pneumonia.

Each folder includes:

Dataset loading and preprocessing steps.

Training, validation, and evaluation results.

Accuracy and loss curves.

Part 1: Baseline Reproduction (Binary Classification)

This phase re-implements the original CNN-based approach for binary classification (COVID vs Normal) using a compact architecture built in TensorFlow/Keras.

Dataset Used:
COVID-19 Radiography Database (Kaggle)

Model Summary:

Two convolution + pooling blocks

Dense layer with dropout regularization

Sigmoid output for binary classification

Optimizer: Adam, Loss: Binary Cross-Entropy

Training Configuration:

Epochs: 8–10

Image size: 128×128

Batch size: 16

Validation split: 20%

Sample Output:

Accuracy vs Epochs graph

Confusion matrix visualization

Classification report with precision, recall, and F1-score

Part 2: Transfer Learning & 3-Class Extension

This phase introduces two improvements:

A) Stronger Backbone (MobileNetV2)

Initialized with ImageNet weights

Global Average Pooling + Dropout + Dense(1, sigmoid) head

Early stopping and learning rate scheduling added

Fine-tuned top 20 layers for improved generalization

Outcome:

Improved validation accuracy and smoother learning curves compared to the scratch CNN.

B) 3-Class Task: COVID vs Normal vs Viral Pneumonia

Adapted data pipeline for multi-class (class_mode='categorical')

Updated model head to Dense(3, softmax)

Loss: Categorical Cross-Entropy

Metrics: Accuracy, Macro F1-Score, Normalized Confusion Matrix

Outcome:

Demonstrates that the same pipeline generalizes to a more realistic multi-class scenario.

Visualized class-wise confusion matrix for interpretability.
