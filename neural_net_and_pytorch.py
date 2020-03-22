# -*- coding: utf-8 -*-
"""

@author: Samip
"""

"""We used Logistic Regression with Pytorch for classification, but were unable to get accuracy above 87%, hence we will try feedforward Neural network."""


"""Preparing data"""

#Import libraries
import pandas as pd
import numpy as np
import torch
import torchvision
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader


#Prepare data
data = MNIST(root = 'data/', download = True, transform = ToTensor())

#Split 20% randomly for validation
def split_indices(n, val_pct):
    #Determine the size of validation set
    n_val = int(val_pct * n)
    #Create random permutation from 0 to n - 1
    idxs = np.random.permutation(n)
    #Pick first n_val from idxs
    return idxs[n_val:], idxs[:n_val]

#Split data in train and validation
train_indices, val_indices = split_indices(len(data), val_pct = 0.2)

print(len(train_indices), len(val_indices))
print("Sample validation indices: ", val_indices[:15])

#We will create data loaders for randomly sampling the data using batches of data
batch_size = 100

#Training sampler and dataloader
train_sampler = SubsetRandomSampler(train_indices)
train_dl = DataLoader(data, batch_size, sampler = train_sampler)

#Validation sampler and dataloader
val_sampler = SubsetRandomSampler(val_indices)
val_dl = DataLoader(data, batch_size, sampler = val_sampler)

"""MODELLING"""

"""We will create a neural network with 1 hidden layer. Instead of one layer, we will use two layers. First layer will transform to batch_size * 784 while second
will convert it to batch_size * hidden_size. It is then passed to activation function. The result is passed to second (output layer), which transforms output
to batch_size * 10. Adding extra layers help in giving more computing power and solve more complex problems.""" 

import torch.nn.functional as F
import torch.nn as nn

class MNISTModel(nn.Module):
    
    #Feedforward neural network with 1 hidden layer
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        #Hidden layer
        self.linear1 = nn.Linear(in_size, hidden_size)
        #Output layer
        self.linear2 = nn.Linear(hidden_size, out_size)
        
    #Forward 
    def forward(self, x):
        #Flatten image tensors
        x = x.view(x.size(0), -1)
        #Get intermediate output using hidden layer
        out = self.linear1(x)
        #Apply activation function
        out = F.relu(out)
        #Get prediction using output layer
        out = self.linear2(out)
        return out

#Create a sample model
input_size = 784
num_class = 10

model = MNISTModel(input_size, hidden_size = 32, out_size = num_class)

#Check parameters of each layer
for t in model.parameters():
    print(t.shape)
    
#Lets predict output for first 100 images of the dataset
for images, labels in train_dl:
    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)
    print("Loss: ", loss.item())
    break

print("Output shape: ", outputs.shape)
print("Sample outputs: ", outputs[:2].data)

"""Using GPU"""
torch.cuda.is_available()

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

device = get_device()

#Define function to move data to the device
def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking = True)

#Use Device DataLoader to move the data to device and wrap existing data loaders
    
class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        #Yield batch of dataafter moving to the data
        for b in self.dl:
            yield to_device(b, self.device)
    
    def __len__(self):
        #Number of devices
        return len(self.dl)
    
#Wrap the data loaders
train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)

for x, y in val_dl:
    print(x.device)
    print(y)
    break


"""Training the model"""
def loss_batch(model, loss_fn, x, y, opt = None, metric = None):
    
    #Generate predictions
    pred = model(x)
    #Calculate loss
    loss = loss_fn(pred, y)
    
    if opt is not None:
        #Compute gradients
        loss.backward()
        #Update parameters
        opt.step()
        #Reset gradients
        opt.zero_grad()
    
    metric_result = None
    if metric is not None:
        #Compute metrics
        metric_result = metric(pred, y)
        
    return loss.item(), len(x), metric_result

#Function to calculate overall loss
def evaluate(model, loss_fn, valid_dl, metric = None):
    with torch.no_grad():
        #Pass each batch thriugh the model
        results = [loss_batch(model, loss_fn, x, y, metric = metric) for x, y in val_dl]
        #Separate loss, counts, and metrics
        losses, nums, metrics = zip(*results)
        #Total size of data
        total = np.sum(nums)
        #Average loss across batches
        avg_loss = np.sum(np.multiply(losses, nums)) / total
        avg_metric = None
        if metrics is not None:
            #Average metrics across batches
            avg_metric = np.sum(np.multiply(metrics, nums)) / total
    return avg_loss, total, avg_metric

#Define function to fit the model
def fit(epochs, lr, model, loss_fn, train_dl, val_dl, metric = None, opt_fn = None):
    losses, metrics = [], []
    
    #Create instance of optimizer
    if opt_fn is None:
        opt_fn = torch.optim.SGD
    opt = torch.optim.SGD(model.parameters(), lr = lr)
    
    for epoch in range(epochs):
        
        #Training
        for x, y in train_dl:
            loss, _, _ = loss_batch(model , loss_fn ,x, y, opt)
            
        #Evaluation
        result = evaluate(model, loss_fn, val_dl, metric)
        val_loss, total, val_metric= result
        
        #Record the loss and metrics
        losses.append(val_loss)
        metrics.append(val_metric)
        
        #Print progress
        if metric is None:
            print("Epoch: [{} / {}], Loss: {:.4f}".format(epoch + 1, epochs, val_loss))
        else:
            print("Epoch: [{} / {}], Loss: {:.4f}, {}: {:.4f}".format(epoch + 1, epochs, val_loss, metric.__name__, val_metric))
        
    return losses, metrics

#Define accuracy function for overall model performance
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim = 1)
    return torch.sum(preds == labels).item() / len(preds)

#Model
model = MNISTModel(input_size, hidden_size = 32, out_size = num_class)
to_device(model, device)

#Evaluate model
val_loss, total, val_acc = evaluate(model, F.cross_entropy, val_dl, accuracy)
print("Loss: {:.4f}, Accuracy: {:.4f}".format(val_loss, val_acc))

"""Initial accuracy is low. We will run for multiple epochs"""
losses1, metrics1 = fit(5, 0.5, model, F.cross_entropy, train_dl, val_dl, accuracy)
losses2, metrics2 = fit(5, 0.1, model, F.cross_entropy, train_dl, val_dl, accuracy)

#View epoch vs accuracy
import matplotlib.pyplot as plt
accuracies = [val_acc] + metrics1 + metrics2
plt.plot(accuracies, '-x')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuraccy vs Epochs")