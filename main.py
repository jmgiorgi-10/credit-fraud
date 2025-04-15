
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from mlp import MLP
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


batch_size = 64
learning_rate = 0.001
num_epochs = 25

# It contains only numerical input variables which are the result of a PCA transformation. 
# Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. 
# Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. 
# Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. 
# The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-sensitive learning. 
# Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise. 

# 1. Multi-layer perceptron
# 2. Variational autoencoder

# class for loading a .csv dataset
class MyDataset(Dataset):

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        return sample, label
    
def train_one_epoch(train_dataloader, device, optimizer, model, loss):

    epoch_loss = 0
    for batch_idx, (inputs, targets) in enumerate(tqdm(train_dataloader)):

        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        input_loss = loss(outputs, targets)
        input_loss.backward()
        optimizer.step()

        epoch_loss += input_loss.item()

    print(f"Epoch Loss: {epoch_loss:.4f}")


def evaluate(test_dataloader, model, device):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in tqdm(test_dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted_labels = torch.max(outputs, 1)

            all_preds.extend(predicted_labels.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    report = classification_report(all_targets, all_preds, target_names=["Not Fraud", "Fraud"], digits=4)
    print("\nClassification Report:\n")
    print(report)


def main():

    mlp = MLP()
    device = torch.device('cpu')
    model = mlp.to(device)

    credit_df = pd.read_csv("../creditcard.csv")
    data_df = credit_df.drop(columns=['Time','Amount','Class'])
    class_df = credit_df['Class']

    import pdb; pdb.set_trace()

    X_train, X_test, y_train, y_test = train_test_split(data_df, class_df, test_size=0.3, random_state=42)
    X_train = torch.tensor(X_train.values, dtype=torch.float32)
    X_test = torch.tensor(X_test.values, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.long)
    y_test = torch.tensor(y_test.values, dtype=torch.long)


    train_dataset = MyDataset(X_train, y_train)
    test_dataset = MyDataset(X_test, y_test)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    loss = nn.CrossEntropyLoss()

    # set model to training mode, and select desired optimizer #
    mlp.train()
    epoch_loss = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

    # training loop #

    for epoch in range(num_epochs):
        train_one_epoch(train_dataloader, device, optimizer, model, loss)

    # Evaluation
    result_dict = evaluate(test_dataloader, model, device)
    # print(f"accuracy: {result_dict['accuracy']}")

    # for batch_idx, (inputs, targets) in enumerate(tqdm(train_dataloader)):

    #     inputs = inputs.to(device)
    #     targets = targets.to(device)

    #     optimizer.zero_grad()
    #     outputs = model(inputs)

    #     input_loss = loss(outputs, targets)
    #     input_loss.backward()
    #     optimizer.step()

    #     epoch_loss += input_loss.item()

    # epoch loss
  
if __name__ == '__main__':
    main()

    