#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 14:44:23 2021

@author: marco
"""
import torchvision.transforms as transforms
import torchvision
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt




def weights_init(m):
    if type(m) == nn.Linear:
        m.weight.data.normal_(0.0, 1e-3)
        m.bias.data.fill_(0.)

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


#-------------------------------------------------
# Calculate the model size (Q1.b)
# if disp is true, print the model parameters, otherwise, only return the number of parameters.
#-------------------------------------------------
def PrintModelSize(model, disp=True):
    #################################################################################
    # TODO: Implement the function to count the number of trainable parameters in   #
    # the input model. This useful to track the capacity of the model you are       #
    # training                                                                      #
    #################################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    model_sz = np.nan
    
    model_sz = np.sum([p.numel() for p in model.parameters() if p.requires_grad])
    
    if disp:
        print("Model size | Number of parameters: ", model_sz)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return model_sz



#-------------------------------------------------
# Calculate the model size (Q1.c)
# visualize the convolution filters of the first convolution layer of the input model
#-------------------------------------------------
def VisualizeFilter(model, path = "", save_to_disk = False, prefix = ""):
    #################################################################################
    # TODO: Implement the functiont to visualize the weights in the first conv layer#
    # in the model. Visualize them as a single image of stacked filters.            #
    # You can use matlplotlib.imshow to visualize an image in python                #
    #################################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    layer_0 = model.layers[0]
    print(layer_0)
    
    weights = torch.Tensor.cpu(layer_0.weight).detach().numpy()
    
    num_rows = 8
    num_cols = 16
    
    
    fig, ax = plt.subplots(nrows = num_rows, ncols = num_cols, figsize = (15, 10))
    fig.suptitle(prefix + "Convolutional layer #0 | Weight visualization")
    
    weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weights)) # Normalize into [0, 1]

    index_weight = 0
    
    for i in range(num_rows):
        for j in range(num_cols):
            
            weights_ij = weights[index_weight, ...]
            
            ax[i, j].imshow(weights_ij)
            ax[i, j].set_xticklabels([])
            ax[i, j].set_yticklabels([])
            index_weight += 1    
        
    if save_to_disk:
         #fig.imsave(path)
        fig.savefig(path)   # save the figure to file
        plt.close(fig)    # close the figure window
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****




class ConvNet(nn.Module):
    def __init__(self, input_size, hidden_layers, num_classes, norm_layer=None):
        super(ConvNet, self).__init__()
        #################################################################################
        # TODO: Initialize the modules required to implement the convolutional layer    #
        # described in the exercise.                                                    #
        # For Q1.a make use of conv2d and relu layers from the torch.nn module.         #
        # For Q2.a make use of BatchNorm2d layer from the torch.nn module.              #
        # For Q3.b Use Dropout layer from the torch.nn module.                          #
        #################################################################################
        layers = []
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        #print("Hidden layers: ", hidden_layers)
        
        layers.append(nn.Conv2d(3, hidden_layers[0], kernel_size = 3, padding = 1, stride = 1))
        
        if norm_layer: # Batch normalization
            layers.append(nn.BatchNorm2d(hidden_layers[0]))
            
        
        layers.append(nn.MaxPool2d(kernel_size = 2, stride = 2))
        layers.append(nn.ReLU())
                
        for i in range(1, len(hidden_layers)): # The -1 since the last layer is Linear
            
            layers.append(nn.Conv2d(hidden_layers[i-1], hidden_layers[i], kernel_size = 3, padding = 1, stride = 1))
            
            if norm_layer:
                layers.append(nn.BatchNorm2d(hidden_layers[i]))
            
            layers.append(nn.MaxPool2d(kernel_size = 2, stride = 2))
            layers.append(nn.ReLU(inplace = False))
                    
        
        layers.append(nn.Flatten())
        
        
        layers.append(nn.Linear(hidden_layers[-1], num_classes))
        layers.append(nn.Dropout(p = 0.2))  # the value of "p" should be between 0.1 and 0.9
        
        
        
        self.layers = nn.Sequential(*layers)
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    def forward(self, x):
        #################################################################################
        # TODO: Implement the forward pass computations                                 #
        #################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        out = self.layers(x)
        
        '''
        # THIS IS JUST FOR DEBUGGING the ouput dims of each layer - hence the assert 1==0 to force a stop, 
        out = x
        
        
        for i in range(0, len(self.layers)):
            out = self.layers[i](out)
            print("i: ", i, " | Layers: ", self.layers[i], " | Out shape: ", out.shape)
        
        
        assert 1 == 0, "STOP HERE"
        '''
    
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return out



def get_dataset_loaders(data_aug_transforms, batch_size, num_training = 49000, num_validation = 1000, download = False):
  
    norm_transform = transforms.Compose(data_aug_transforms+[transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                         ])
    test_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                         ])
    cifar_dataset = torchvision.datasets.CIFAR10(root='datasets/',
                                               train=True,
                                               transform=norm_transform,
                                               download= download)
    
    test_dataset = torchvision.datasets.CIFAR10(root='datasets/',
                                              train=False,
                                              transform=test_transform
                                              )
    
    
    
    mask = list(range(num_training))
    train_dataset = torch.utils.data.Subset(cifar_dataset, mask)
    mask = list(range(num_training, num_training + num_validation))
    val_dataset = torch.utils.data.Subset(cifar_dataset, mask)
    
    
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                               batch_size=batch_size,
                                               shuffle=False)
    
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)
    
    return train_loader, val_loader, test_loader






def complete_training_and_validation(model, num_epochs, train_loader, val_loader, device, learning_rate, learning_rate_decay, reg, batch_size,
                                     criterion = nn.CrossEntropyLoss(), save_models = False):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=reg)
    
    # Train the model
    lr = learning_rate
    total_step = len(train_loader)
    loss_train = []
    loss_val = []
    accuracy_val = []
       
    best_model_accuracy = np.nan
    early_stopped_accuracy = np.nan
    best_model = None
    early_stopped_model = None
    is_early_stopped = False

    for epoch in range(num_epochs):
    
        model.train()
    
        loss_iter = 0
        for i, (images, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)
    
            # Forward pass
            outputs = model(images)
            
            loss = criterion(outputs, labels)
    
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            loss_iter += loss.item()
            
            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                       .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
                
        loss_train.append(loss_iter/(len(train_loader)*batch_size))
    
        
        # Code to update the lr
        lr *= learning_rate_decay
        update_lr(optimizer, lr)
        
            
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            loss_iter = 0
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                loss = criterion(outputs, labels)
                loss_iter += loss.item()
            
            loss_val.append(loss_iter/(len(val_loader)*batch_size))
    
            accuracy = 100 * correct / total
            accuracy_val.append(accuracy)
            
            print('Validation accuracy is: {} %'.format(accuracy))
            
            
            epsilon = 1e-4 # Difference which is allowed
            
            
            DEFAULT_PATIENCE = 4

            if num_epochs > 10:
              DEFAULT_PATIENCE = 8

            DEFAULT_EARLY_STOPPED_MODEL_PATH = './early_stopped_model.bin'
            DEFAULT_BEST_MODEL_PATH = './best_model.bin'
            
            # Check the last n accuracies and determine if it's the case to stop
            
    
            if accuracy >= np.max(accuracy_val): # Check if the current validation accuracy is the best among the previous ones
                best_model = model
                best_model_accuracy = accuracy
                
                if save_models:
                    torch.save(best_model, DEFAULT_BEST_MODEL_PATH)
                    print("Best model saved. Best accuracy: ", accuracy)
        
            
            if DEFAULT_PATIENCE < len(accuracy_val): # Only start early stopping when you have enough accuracies
                
                last_n_accuracies = accuracy_val[-DEFAULT_PATIENCE - 1:] # Get the last nth accuracies 
                last_n_accuracies = last_n_accuracies[:-1]
    
                patience = DEFAULT_PATIENCE # Just for clarity we define new names
                current_accuracy = accuracy # also here, just for clarity
    
                print("Last n accuracies: ", last_n_accuracies)
                print("Current accuracy: ", current_accuracy)        
    
                for prev_accuracy in last_n_accuracies:
                    
                    if (current_accuracy - epsilon) < prev_accuracy: # We have an improvement!
                        patience -= 1
                        print("The current accuracy is still too low | Accuracy: ", current_accuracy, " | Patience left: ", patience)
    
                    
                    
                if patience == 0: # No improvements in the last n iterations
                    early_stopped_model = model  #Save the best model
                    early_stopped_accuracy = current_accuracy
                    is_early_stopped = True
                    if save_models:
                        torch.save(early_stopped_model, DEFAULT_EARLY_STOPPED_MODEL_PATH)
                        print("Found the early stopped model! Accuracy: ", current_accuracy)
                    
                    break
                
                
    
    return best_model, early_stopped_model, loss_train, loss_val, best_model_accuracy, early_stopped_accuracy, is_early_stopped, accuracy_val
            



def test_model(model, test_loader, device):
    
    with torch.no_grad():
        correct = 0
        total = 0
        
        for images, labels in test_loader:
            
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if total == 1000:
                break
            
    test_accuracy =  100 * correct / total
    
    return test_accuracy



