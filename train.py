# librairies importation
import pandas as pd
import numpy as np

import torch
from torch import nn, optim
from torch.optim import lr_scheduler

import torchvision
from torchvision import datasets, transforms, models

from collections import OrderedDict
from os import listdir
import time
import copy
import argparse
import json


#Variables init
arch = 'vgg16'
hidden_units = 4096
learning_rate = 0.001
epochs = 3
device = 'cpu'

# Set up parameters for entry in command line
parser = argparse.ArgumentParser()
parser.add_argument('data_dir',type=str, help='Location of directory with data for image classifier to train and test',default='./flowers')
parser.add_argument('-a','--arch',action='store',type=str, help='Choose among 3 pretrained networks - vgg16, alexnet, and densenet121',default='vgg16')
parser.add_argument('-H','--hidden_units',action='store',type=int, help='Select number of hidden units for 1st layer',default=4096)
parser.add_argument('-l','--learning_rate',action='store',type=float, help='Choose a float number as the learning rate for the model',default=0.001)
parser.add_argument('-e','--epochs',action='store',type=int, help='Choose the number of epochs you want to perform gradient descent',default=3)
parser.add_argument('-s','--save_dir',action='store', type=str, help='Select name of file to save the trained model', default="./checkpoint.pth")
parser.add_argument('-g','--gpu',action='store_true',help='Use GPU if available', default="gpu")

args = parser.parse_args()

# Select parameters entered in command line
if args.arch:
    arch = args.arch
if args.hidden_units:
    hidden_units = args.hidden_units
if args.learning_rate:
    learning_rate = args.learning_rate
if args.epochs:
    epochs = args.epochs
if args.save_dir:
    save_dir = args.save_dir
if args.gpu:        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_dir = args.data_dir + '/train'
valid_dir = args.data_dir + '/valid'
test_dir = args.data_dir + '/test'

# TODO: Define your transforms for the training, validation, and testing sets
data_transforms ={
    'data_train_transforms':transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])]),
   
    'data_test_transforms' : transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])]),

    'data_validation_transforms' : transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])}



# TODO: Load the datasets with ImageFolder
image_datasets = {'train_data' : datasets.ImageFolder(train_dir, transform=data_transforms['data_train_transforms']),
    'test_data' : datasets.ImageFolder(test_dir, transform=data_transforms['data_test_transforms']),
    'valid_data' : datasets.ImageFolder(valid_dir, transform=data_transforms['data_validation_transforms']),
                }


# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders = {'trainloader' : torch.utils.data.DataLoader(image_datasets['train_data'], batch_size=64, shuffle=True),
    'testloader' : torch.utils.data.DataLoader(image_datasets['test_data'], batch_size=32),
    'validloader' : torch.utils.data.DataLoader(image_datasets['valid_data'], batch_size=32)}

with open('cat_to_name.json', 'r') as f:
    labels_struc = json.load(f)


model = getattr(models,arch)(pretrained=True)


for param in model.parameters():
    param.requires_grad = False

in_features = model.classifier[0].in_features
model.classifier =   nn.Sequential(OrderedDict([
                           ('fc1',nn.Linear(in_features,hidden_units)),
                           ('ReLu1',nn.ReLU()),
                           ('Dropout1',nn.Dropout(p=0.2)),
                           ('fc2',nn.Linear(hidden_units,1024)),
                           ('ReLu2',nn.ReLU()),
                           ('Dropout2',nn.Dropout(p=0.2)),
                           ('fc3',nn.Linear(1024,102)),
                           ('output',nn.LogSoftmax(dim=1))
]))

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

model.to(device);

steps = 0
running_loss = 0
print_every = 40
for epoch in range(epochs):
    for inputs, labels in dataloaders['trainloader']:
        steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if steps % print_every == 0:
            validation_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in dataloaders['validloader']:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    validation_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                   
            print("Epoch: {}/{}.. ".format(epoch+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/len(dataloaders['trainloader'])),
                  "validation Loss: {:.3f}.. ".format(validation_loss/len(dataloaders['validloader'])),
                  "validation Accuracy: {:.3f}%".format(accuracy/len(dataloaders['validloader']) *100))
                  
            running_loss = 0
            model.train()

#save our checkpoint            
model.class_to_idx = image_datasets['train_data'].class_to_idx
torch.save(model,save_dir)