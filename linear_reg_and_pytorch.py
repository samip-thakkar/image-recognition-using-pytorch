# -*- coding: utf-8 -*-
"""

@author: Samip
"""

"""We’ll create a model that predicts crop yields for apples and oranges (target variables) by looking at 
the average temperature, rainfall and humidity (input variables or features) in a region.

In a linear regression model, each target variable is estimated to be a weighted sum of the input variables, offset by some constant, known as a bias.
It means that the yield of apples is a linear or planar function of temperature, rainfall or humidity"""

import numpy as np
import torch

#Training data (Columns: Temperature, Rainfall, Humidity)
inputs = np.array([[73, 67, 43], 
                   [91, 88, 64], 
                   [87, 134, 58], 
                   [102, 43, 37], 
                   [69, 96, 70]], dtype='float32')

# Targets (apples, oranges)
targets = np.array([[56, 70], 
                    [81, 101], 
                    [119, 133], 
                    [22, 37], 
                    [103, 119]], dtype='float32')


#Convert the inputs and targets to tensors
inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)

"""Linear Regression model from scratch"""

#Define weights as w, and bias as b both as matrices The first row of w and first element of b is used for predicting apples, while second for oranges.
w = torch.randn(2, 3, requires_grad = True)
b = torch.randn(2, requires_grad = True)

#Define the model
def model(x):
    return x @ w.t() + b        # @ - Matrix multiplication, .t() - Transpose

#Generate predictions
pred = model(inputs)
print(pred)

#Compare predictions with actual targets
print(targets)

"""There is huge difference as we have randomly assigned the weights and biases."""

#Evaluate model using loss function (MSE loss).
def mse(pred, target):
    dif = pred - target
    return torch.sum(dif * dif) / dif.numel()       #numel - Number of elements

#Compute the loss
loss = mse(pred, targets)
print("Loss: ", loss)


"""Compute the gradients"""

#We can compute gradients using backward method for varuables whose requires_grad is True.
loss.backward()

#Gradient of a variable can be obtained by .grad property
print(w)
print(w.grad)


"""The increase or decrease in loss by changing a weight element is proportional to the value of the gradient of the loss w.r.t. that element.
Before we proceed, we reset the gradients to zero by calling .zero_() method. We need to do this, because PyTorch accumulates, gradients i.e. the next time we call .backward on the loss,
the new gradient values will get added to the existing gradient values, which may lead to unexpected results."""

w.grad.zero_()
b.grad.zero_()

print(w.grad)
print(b.grad)

"""Adjust the weights and bias using gradient descent. Gradient descent optimization algorithm works in following steps: Generate predictions, calculate loss,
compute gradients (w.r.t. w and b), adjust weights by subtracting small quantity proportional to gradient, reset gradient to zero"""

#Generate prediction
pred = model(inputs)
print(pred)

#Calculate loss
loss = mse(pred, targets)
print(loss)

#Computer gradients
loss.backward()
print(w.grad)
print(b.grad)

#We update the weights and biases using gradients computed.
with torch.no_grad():       #torch.no_grad tells Pytorch that we won't track, modify or calculate gradients while updating the weights and biases
    w -= w.grad * 1e-5      # 1e-5 is also called learning rate.
    b -= b.grad * 1e-5

#Reset gradients to zero
w.grad.zero_()
b.grad.zero_()

"""Repeat the process with new weights and bias"""
pred = model(inputs)
loss = mse(pred, targets)
print(loss)

#We can see that loss dropped from 16745 to 11681

"""Train with multiple epochs (100 here)"""
for i in range(100):
    pred = model(inputs)
    loss = mse(pred, targets)
    loss.backward()
    with torch.no_grad():
        w -= w.grad * 1e-5
        b -= b.grad * 1e-5
        w.grad.zero_()
        b.grad.zero_()

#Verify the loss now
pred = model(inputs)
loss = mse(pred, targets)
print(loss)

""" Linear Regression using Pytorch in-builts"""

#We will use torch.nn for in built neural networks facility
import torch.nn as nn
# Input (temp, rainfall, humidity)
inputs = np.array([[73, 67, 43], [91, 88, 64], [87, 134, 58], 
                   [102, 43, 37], [69, 96, 70], [73, 67, 43], 
                   [91, 88, 64], [87, 134, 58], [102, 43, 37], 
                   [69, 96, 70], [73, 67, 43], [91, 88, 64], 
                   [87, 134, 58], [102, 43, 37], [69, 96, 70]], 
                  dtype='float32')

# Targets (apples, oranges)
targets = np.array([[56, 70], [81, 101], [119, 133], 
                    [22, 37], [103, 119], [56, 70], 
                    [81, 101], [119, 133], [22, 37], 
                    [103, 119], [56, 70], [81, 101], 
                    [119, 133], [22, 37], [103, 119]], 
                   dtype='float32')

inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)

#Create tensordata
from torch.utils.data import TensorDataset #TensorDataset allows us to access the small portion of the training data using indexing notations.

#Define dataset
train_data = TensorDataset(inputs, targets)

#Create DataLoader
from torch.utils.data import DataLoader   #DataLoader helps in split the data into batches. We can shuffle and random sample data using it.

#Define DataLoader
batch_size = 5
train_dl = DataLoader(train_data, batch_size, shuffle = True)

#Initialize the weights using nn.Linear instead of setting manually

#Define model
model = nn.Linear(3,2)
print(model.weight)
print(model.bias)

#Generate the predictions
pred = model(inputs)

#Loss funtion. We will use built in mse_loss function
import torch.nn.functional as F
loss_fn = F.mse_loss

#Compute loss
loss = loss_fn(pred, targets)
loss

#OPTIMIZER
#We will use optimizer.SGD for optimization
opt = torch.optim.SGD(model.parameters(), lr = 1e-5)

#TRAIN THE MODEL
# Utility function to train the model
def fit(num_epochs, model, loss_fn, opt):
    
    # Repeat for given number of epochs
    for epoch in range(num_epochs):
        
        # Train with batches of data
        for xb,yb in train_dl:
            
            # 1. Generate predictions
            pred = model(xb)
            
            # 2. Calculate loss
            loss = loss_fn(pred, yb)
            
            # 3. Compute gradients
            loss.backward()
            
            # 4. Update parameters using gradients
            opt.step()
            
            # 5. Reset the gradients to zero
            opt.zero_grad()
        
        # Print the progress for every 10th epoch
        if (epoch+1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
            
fit(100, model, loss_fn, opt)

pred = model(inputs)

loss = loss_fn(pred, targets)
print(loss)