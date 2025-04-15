# Credit Card Fraud Detection using MLP

This project implements a Multi-Layer Perceptron (MLP) model in PyTorch to detect fraudulent credit card transactions using a publicly available dataset. The dataset contains transactions made by European cardholders, with only ~0.17% of the entries labeled as fraud — resulting in a highly imbalanced classification problem.

## Dataset Overview

The dataset consists of 284,807 transactions. Each entry includes 28 PCA-transformed features (V1 to V28), along with Time, Amount, and the target variable Class (0 = Not Fraud, 1 = Fraud).For this implementation, we exclude Time and Amount and focus on the normalized features.

## Model

Multilayer-perceptron (two hidden layers of dimension 80, and ReLU activation).
Dropout layers used between layers to randomly deactivate neurons during training, preventing the model from becoming too reliant on specific feature combinations. This is useful because fraudulent patterns are sparse and subtle. We used L2 regularization (via the weight_decay parameter in the Adam optimizer) to penalize large weights and encourage the model to generalize better, helping mitigate overfitting to the dominant class (non-fraud).

## Results

| Class      | Precision | Recall | F1-score | Support |
|------------|-----------|--------|----------|---------|
| Not Fraud  | 0.9997    | 0.9998 | 0.9997   | 85,307  |
| Fraud      | 0.8485    | 0.8235 | 0.8358   |   136   |
| **Accuracy** |           |        | **0.9995** | 85,443  |
| **Macro Avg** | 0.9241    | 0.9116 | 0.9178   | 85,443  |
| **Weighted Avg** | 0.9995 | 0.9995 |

## Prerequisites

```bash
pip install torch
```

## Future Contributions
Consider Time and Amount transaction data, and perform hyperparameter grid-search (hidden layer size, network depth, batch size).
