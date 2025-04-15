
import torch
import torch.nn as nn

class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(28, 80) # linear transformation 1
        self.linear2 = nn.Linear(80, 80) # linear transformation 2
        self.linear3 = nn.Linear(80, 2)   # linear transformation 3

        self.activation = nn.ReLU() # activation function
        self.dropout = nn.Dropout(p=0.5)  # 50% dropout rate (you can tune this)

    def forward(self, data):

    #    
        x = self.linear1(data)
        x = self.activation(x)
        x = self.dropout(x)

        y = self.linear2(x)
        y = self.activation(y)
        y = self.dropout(y)

        z = self.linear3(y)
      
        return z

# class VAE(nn.Module):

#     def __init__(self):

#         super(VAE, self).__init__()

#         self.inputToLatent = nn.Linear(20, 10)
#         self.latentToOutput = nn.Linear(10, 20)

#     def forward(self, x):

#         # right now this is a simple autoencoder, without any priors, which is equivalent to principle component analysis.

#         z = x @ self.inputToLatent
#         out = z @ self.latentToOutput

#         return z



