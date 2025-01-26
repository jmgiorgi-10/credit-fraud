**Credit card fraud detection**
Using 28 principle components provided in the kaggle dataset, and provided fraud labels.

**Model**
Multilayer-perceptron (two hidden layers of dimension 80, and ReLU activation), achieving >99% accuracy, with 70/30% test-train split. 

**Dataset**
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data. 

**Prerequisites**
```bash
pip install torch
```

**Future Contributions**
Consider Time and Amount transaction data, and perform hyperparameter grid-search (hidden layer size, network depth, batch size).
