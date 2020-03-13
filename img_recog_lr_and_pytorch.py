# -*- coding: utf-8 -*-
"""

@author: Samip
"""

#Import necessary libraries
import torch
from torchvision.datasets import MNIST

#Download the training dataset
df = MNIST(root = 'data/', download = True)
print("Total number of images for training: %s" %len(df))

#Download the test dataset
df_test = MNIST(root = 'data/', train = False)
print("The total number of images for testing: %s" %len(df_test))

#View a sample image
import matplotlib.pyplot as plt
image, label = df[10]    
plt.imshow(image)

"""As due to low image quality, pytorch doesn't know how to work with the images, so we will convert images to tensors"""
import torchvision.transforms as transforms
#MNIST data images to tensors
df = MNIST(root = 'data/', train = True, transform = transforms.ToTensor())

img_tensor, label = df[0]
print("Tensor shape: %s, Label: %s" %(img_tensor.shape, label))

"""Now the image is converted to 1 x 28 x 28 tensor. First dimension is used for color. As MNIST images are greyscale, they have only one channel
Other images have three channels, Red, Blue and Green."""
print("0 represents black and 1 as white")
print(img_tensor[:, 10:15, 10:15])
print(torch.max(img_tensor), torch.min(img_tensor))

#Plot the tensor as image 
plt.imshow(img_tensor[0], cmap = 'gray')


#Training and validate the datasets
#Function to randomly pick given fraction into validation set
import numpy as np
def split_indices(n, val_pct):
    #Determine the size of validation set
    n_val = int(n * val_pct)
    #Create random permutations from 0 to n - 1
    idxs = np.random.permutation(n)
    #Pick first n_val indices for validation set
    return idxs[n_val:], idxs[:n_val]

"""As the training dataset is in order of target label, it is necessary to shuffle before training the model"""
train_indices, val_indices = split_indices(len(df), 0.2)
print(len(train_indices), len(val_indices))
print("Sample validation indices: ", val_indices[:20])

"""We have set validation set to 20% and they are shuffled. PyTorch data loaders for each of these using a SubsetRandomSampler, 
which samples elements randomly from a given list of indices, while creating batches of data."""

from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader

batch_size = 100

#Train sampler data and data loader
train_sampler = SubsetRandomSampler(train_indices)
train_loader = DataLoader(df, batch_size, sampler = train_sampler)

#Validating sampler data and data loader
val_sampler = SubsetRandomSampler(val_indices)
val_loader = DataLoader(df, batch_size, sampler = val_sampler)

"""DEFINE MODEL: LOGISTIC REGRESSION"""

import torch.nn as nn
input_size = 28 * 28
num_class = 10

#Logistic Regression model
lr = nn.Linear(input_size, num_class)

print(lr.weight.shape)
print(lr.bias.shape)

#Generate outputs using our model
"""Our images are of shape 1 x 28 x 28, while we need a vector of size 784, hence we need to flatten it out by using reshape."""
"""Inside __init__, we initialize the weights and biasesusing nn.Linear. When we pass batch of inputs to model in forward method,
we flatten out the output tensor and then pass it to self.linear. xb.reshape has two dimensions: first is -1 that lets PyTorch automatically set 
dimension based on shape of original tensor. 2nd dimension is 28 * 28, which is 784.""" 

class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size, num_class)
    
    def forward(self, x):
        x = x.reshape(-1, 784)
        out = self.linear(x)
        return out

model = MNISTModel()


"""Output of above method is parameter method, which returns a list containing weights and biases."""
print(model.linear.weight.shape, model.linear.bias.shape)

#Testing our model
for images, labels in train_loader:
    outputs = model(images)
    break

print('Output shape: ', outputs.shape)
print('Sample outputs: ', outputs[:2].data)


"""We have 100 images and 10 classes. Our aim was to show the output as  probabilities such that values lie in (0, 1), and the sum for 
each image(row) is 1, which is not the case. To convert the output to possibilities, we use softmax(activation) function."""

import torch.nn.functional as F

#Apply softmax for each output row
probs = F.softmax(outputs, dim = 1)

#Check sample probabilities
print("Sample probabilities: ", probs[:2].data)
#Add up the probabilities of an output row
print("Sum of probabilities: " ,torch.sum(probs[0]).item())

"""Finaly we can determine the predicted label for each image by simply choosing index with highest probability in each row. 
This can be done using torch.max, which returns largest element and its index with particular dimension."""

max_probs, preds = torch.max(probs, dim = 1)
print("Predicted labels: ", preds)
#Check with actual labels
print("Actual labels: ", labels)
#Print the maximum probability
print("Maximum probability: ", max_probs)

"""They are all different because we randomly initialized the weights and biases. We need to train the model, i.e. adjust the weight
using gradiant descent to make better predictions."""

#Calculate accuracy
def accuracy(l1, l2):
    return torch.sum(l1 == l2).item()/ len(l1)

print("Accuracy: ", accuracy(preds, labels))

"""We can't use accuracy as a measure for loss function for optimizing gradient descent, it may be good for classification, 
but not for the evaluation of loss function. It is because it is non-deferrentiable and it can't provide a feedback for increment."""
    
loss_fn = F.cross_entropy

#Loss function for current batch of data
loss = loss_fn(outputs, labels)
print("Loss: ", loss)

"""OPTIMIZER: We will use optim.SGD(Stochastic Gradient Descent) to update weights and biases during the training."""
learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


"""Training the model"""

#We will define a function loss_batch which will calculate loss of batch of data, optimally perform gradient descent, and calulate the accuracy
def loss_batch(model, loss_fn, x, y, opt=None, metric=None):
    # Calculate loss
    preds = model(x)
    loss = loss_fn(preds, y)
                     
    if opt is not None:
        # Compute gradients
        loss.backward()
        # Update parameters             
        opt.step()
        # Reset gradients
        opt.zero_grad()
    
    metric_result = None
    if metric is not None:
        # Compute the metric
        metric_result = metric(preds, y)
    
    return loss.item(), len(x), metric_result

#We define function evaluate which will calculate the overall lose of validation function
def evaluate(model, loss_fn, valid_dl, metric=None):
    with torch.no_grad():
        # Pass each batch through the model
        results = [loss_batch(model, loss_fn, xb, yb, metric=metric)
                   for xb,yb in valid_dl]
        
        # Separate losses, counts and metrics
        losses, nums, metrics = zip(*results)
        
        # Total size of the dataset
        total = np.sum(nums)
        
        # Avg. loss across batches 
        avg_loss = np.sum(np.multiply(losses, nums)) / total
        avg_metric = None
        
        if metric is not None:
            # Avg. of metric across batches
            avg_metric = np.sum(np.multiply(metrics, nums)) / total
    
    return avg_loss, total, avg_metric


#We will define accuracy function to operate on entire batch directly and use as a metric while fitting
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.sum(preds == labels).item() / len(preds)

#Lets check how well the model performs on validation set with initial weight and biases
val_loss, total, val_acc = evaluate(model, loss_fn, val_loader, metric=accuracy)
print('Loss: {:.4f}, Accuracy: {:.4f}'.format(val_loss, val_acc))

#Initial accuracy is very low as expected (7%)

#We will define a function fit using loss_batch and evaluate
def fit(epochs, model, loss_func, opt, train_dl, val_dl, metric = None):
    for epoch in range(epochs):
        
        #Training
        for x, y in train_dl:
            loss, _, _ = loss_batch(model, loss_func, x, y, opt)
        
        #Evaluation
        result = evaluate(model, loss_func, val_dl, metric)
        val_loss, total, val_metric = result
        
        #Print the results
        if metric is None:
            print('Epoch [{} / {}], Loss = {:.4f}'.format(epoch + 1, epochs, val_loss))
        else:
            print('Epoch [{} / {}], Loss = {:.4f}, {} : {:.4f}'.format(epoch + 1, epochs, val_loss, metric.__name__, val_metric))
            
#Lets train model on 5 epoch and see the results
model = MNISTModel()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
fit(5, model, F.cross_entropy, optimizer, train_loader, val_loader, accuracy)

"""Plot accuracy to epochs graph"""
accuracies = [0.1128, 0.6590, 0.7448, 0.7761, 0.7925, 0.8049,
              0.8149, 0.8218, 0.8270, 0.8330, 0.8374, 
              0.8407, 0.8437, 0.8472, 0.8498, 0.8526,
              0.8551, 0.8572, 0.8591, 0.8607, 0.8617]

plt.plot(accuracies, '-x')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Accuracy vs. No. of epochs');


"""Testing with a test data"""
test_dataset = MNIST(root = 'data/', train = False, transform = transforms.ToTensor())

img, label = test_dataset[111]
plt.imshow(img[0], cmap = 'gray')
print(label)

"""Predict the image"""
def predict_image(img, model):
    x = img.unsqueeze(0)
    y = model(x)
    _, y_pred = torch.max(y, dim = 1)
    return y_pred[0].item()

#View the result
img, label = test_dataset[193]
plt.imshow(img[0], cmap='gray')
print('Label:', label, ', Predicted:', predict_image(img, model))

img, label = test_dataset[1839]
plt.imshow(img[0], cmap='gray')
print('Label:', label, ', Predicted:', predict_image(img, model))

"""Check the overall accuracy of the model"""
test_loader = DataLoader(test_dataset, batch_size = 200)
test_loss, total, test_acc = evaluate(model, loss_fn, test_loader, metric = accuracy)
print('Loss: {:.4f}, Accuracy: {:.4f}'.format(test_loss, test_acc))

"""Save the model so that we can reuse it."""
torch.save(model.state_dict(), 'mnist-logistic.pth')
model.state_dict()

"""Verify the model saved"""
#Create an object and test on same data to verify model
model2 = MNISTModel()
model2.load_state_dict(torch.load('mnist-logistic.pth'))
model2.state_dict()
test_loss, total, test_acc = evaluate(model, loss_fn, test_loader, metric=accuracy)
print('Loss: {:.4f}, Accuracy: {:.4f}'.format(test_loss, test_acc))