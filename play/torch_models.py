"""
This works in tandem with the mlp-mnistDataset-torch notebook based on this (https://www.kaggle.com/code/mishra1993/pytorch-multi-layer-perceptron-mnist/notebook) tutorial
"""

import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    """Define the NN architecture"""

    def __init__(self, hidden_1: int, hidden_2: int, height: int, width: int, dropout_rate: float):
        super(Net, self).__init__()

        self.height = heightght = height
        self.width = width
        # linear layer (784 -> hidden_1)
        self.fc1 = nn.Linear(width * height, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
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

    def train_network(self, number_of_epochs: int, optimizer, criterion_function, train_loader, valid_loader):
        # set initial "min" to infinity
        valid_loss_min = np.Inf

        for epoch in range(number_of_epochs):
            train_loss = 0.0
            valid_loss = 0.0

            ###################
            # train the model #
            ###################
            self.train() # prep model for training
            for data, target in train_loader:
                # clear the gradients of all optimized variables
                optimizer.zero_grad()
                # forward pass to compute predictions, loss, backward pass to compute gradient wrt model params
                output = self(data)
                loss = criterion_function(output, target)
                loss.backward()
                optimizer.step()
                # update running training loss
                train_loss += loss.item() * data.size(0)
            
            ######################    
            # validate the model #
            ######################
            self.eval() # prep model for evaluation
            for data, target in valid_loader:
                # forward pass to compute predictions, loss, update running validation loss
                output = self(data)
                loss = criterion_function(output, target)
                valid_loss += loss.item() * data.size(0)
            
            # print training/validation statistics 
            # calculate average loss over an epoch
            train_loss = train_loss / len(train_loader.dataset)
            valid_loss = valid_loss / len(valid_loader.dataset)
            
            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
                epoch+1, 
                train_loss,
                valid_loss
                ))
            
            # save model if validation loss has decreased
            if valid_loss <= valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))
                torch.save(self.state_dict(), 'nn_model.pt')
                valid_loss_min = valid_loss

    def predict(self, data_loader):
        prediction_list = []
        for i, batch in enumerate(data_loader):
            outputs = self(batch)    
            _, predicted = torch.max(outputs.data, 1)
            prediction_list.append(predicted.cpu())
        
        return prediction_list



