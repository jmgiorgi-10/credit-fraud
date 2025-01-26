**Credit card fraud detection**
Using 28 principle components provided in the kaggle dataset, and provided fraud labels.

**Model**
Credit card fraud detection, with Multilayer-perceptron (two hidden layers of dimension) approach, achieving >99% accuracy, with 70/30% test-train split. Used two hidden layers of dimension 80, and ReLU activation.

**Dataset**
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data. 

**Prerequisites**
```bash
pip install torch
```

**Future Contributions**
Consider Time and Amount transaction data, and perform hyperparameter grid-search (hidden layer size, network depth, batch size).
