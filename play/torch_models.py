"""
This works in tandem with the mlp-mnistDataset-torch notebook based on this (https://www.kaggle.com/code/mishra1993/pytorch-multi-layer-perceptron-mnist/notebook) tutorial
"""

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    """Define the NN architecture"""

    def __init__(self, hidden_1: int, hidden_2: int, width: int, height: int, dropout_rate: float):
        super(Net, self).__init__()

        # linear layer (784 -> hidden_1)
        self.fc1 = nn.Linear(width * height, hidden_1)
        self.fc2 = nn.Linear(width * height, hidden_2)
        self.fc3 = nn.Linear(hidden_2, 10)

        # dropout to prevent overfitting
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        # flatten image
        x = x.view(-1, self.width * self.height)
        
        # add hidden layer, with relu activation function, dropout, relu, dropout, output
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x



