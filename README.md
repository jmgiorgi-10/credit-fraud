**Credit card fraud detection**

**Model**

Credit card fraud detection, with Multilayer-perceptron (two hidden layers of dimension) approach, achieving >99% accuracy, with 70/30% test-train split. I used two hidden layers of dimension 80, with ReLU activation, and used 28 principle components provided in the kaggle dataset. 

**Dataset**
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data. 
For next steps, I will consider Time and Amount transaction data, and perform grid-search for hyperparameters - hidden layer size, network depth, batch size.

**Prerequisites**
'''
pip install pytorch
