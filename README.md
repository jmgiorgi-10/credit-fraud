**Credit card fraud detection**
Vew

**Model**

Credit card fraud detection, with Multilayer-perceptron (two hidden layers of dimension) approach, achieving >99% accuracy, with 70/30% test-train split. I used two hidden layers of dimension 80, with ReLU activation, and used 28 principle components provided in the kaggle dataset. 

**Dataset**
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data. 

**Prerequisites**
```bash
pip install torch

****Future contributions
Consider Time and Amount transaction data, and perform hyperparameter grid-search (hidden layer size, network depth, batch size).
