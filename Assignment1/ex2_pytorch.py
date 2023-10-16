import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

#### For resolving SSL Certification Error #####
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
################################################

def weights_init(m):
    if type(m) == nn.Linear:
        m.weight.data.normal_(0.0, 1e-3)
        m.bias.data.fill_(0.)

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

#--------------------------------
# Device configuration
#--------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: %s'%device)

#--------------------------------
# Hyper-parameters
#--------------------------------
input_size = 32 * 32 * 3

# 2,3,4,5 layers 
History = {}
hiddenSizetest = [
                    # [50],
                  # [512,256],
                  # [1024,512,256],
                  [128,32,32,16],
                  [1024,1024,512,512,256]
                  ]

for NetSize in hiddenSizetest:
    num_classes = 10
    hidden_size = NetSize
    num_epochs = 10
    batch_size = 200
    learning_rate = 1e-3
    learning_rate_decay = 0.95
    reg=0.001
    num_training= 49000
    num_validation =1000
    train = True
    
    #-------------------------------------------------
    # Load the CIFAR-10 dataset
    #-------------------------------------------------
    norm_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                         ])
    cifar_dataset = torchvision.datasets.CIFAR10(root='datasets/',
                                               train=True,
                                               transform=norm_transform,
                                               download=True)
    
    test_dataset = torchvision.datasets.CIFAR10(root='datasets/',
                                              train=False,
                                              transform=norm_transform
                                              )
    #-------------------------------------------------
    # Prepare the training and validation splits
    #-------------------------------------------------
    mask = list(range(num_training))
    train_dataset = torch.utils.data.Subset(cifar_dataset, mask)
    mask = list(range(num_training, num_training + num_validation))
    val_dataset = torch.utils.data.Subset(cifar_dataset, mask)
    
    #-------------------------------------------------
    # Data loader
    #-------------------------------------------------
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                               batch_size=batch_size,
                                               shuffle=False)
    
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)
    
    
    
    #======================================================================================
    # Q4: Implementing multi-layer perceptron in PyTorch
    #======================================================================================
    # So far we have implemented a two-layer network using numpy by explicitly
    # writing down the forward computation and deriving and implementing the
    # equations for backward computation. This process can be tedious to extend to
    # large network architectures
    #
    # Popular deep-learning libraries like PyTorch and Tensorflow allow us to
    # quickly implement complicated neural network architectures. They provide
    # pre-defined layers which can be used as building blocks to define our
    # network. They also enable automatic-differentiation, which allows us to
    # define only the forward pass and let the libraries perform back-propagation
    # using automatic differentiation.
    #
    # In this question we will implement a multi-layer perceptron using the PyTorch
    # library.  Please complete the code for the MultiLayerPerceptron, training and
    # evaluating the model. Once you can train the two layer model, experiment with
    # adding more layers and report your observations
    #--------------------------------------------------------------------------------------
    
    #-------------------------------------------------
    # Fully connected neural network with one hidden layer
    #-------------------------------------------------
    class MultiLayerPerceptron(nn.Module):
        def __init__(self, input_size, hidden_layers, num_classes):
            super(MultiLayerPerceptron, self).__init__()
            #################################################################################
            # TODO: Initialize the modules required to implement the mlp with the layer     #
            # configuration. input_size --> hidden_layers[0] --> hidden_layers[1] .... -->  #
            # hidden_layers[-1] --> num_classes                                             #
            # Make use of linear and relu layers from the torch.nn module                   #
            #################################################################################
            
            layers = [] #Use the layers list to store a variable number of layers
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            hidden_size.insert(0, input_size) 
            hidden_size.append(num_classes) 
            
            for H in range(len(hidden_size)-1):
                layers.append(nn.Linear(hidden_size[H], hidden_size[H+1]))
            
            # layers.append(nn.Linear(hidden_size[0], 50))
            # layers.append(nn.Linear(50, 10))
            
            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
            # Enter the layers into nn.Sequential, so the model may "see" them
            # Note the use of * in front of layers
            self.layers = nn.Sequential(*layers)
    
        def forward(self, x):
            #################################################################################
            # TODO: Implement the forward pass computations                                 #
            # Note that you do not need to use the softmax operation at the end.            #
            # Softmax is only required for the loss computation and the criterion used below#
            # nn.CrossEntropyLoss() already integrates the softmax and the log loss together#
            #################################################################################
            
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            out = x.view(x.size(0), -1)
    
            for layer in self.layers[:-1]:
                out = layer(out)
                out = F.relu(out)
            out = self.layers[-1](out)  
            
            #out = self.layers[0](out)
            #out = F.relu(out)
            #out = self.layers[1](out)
            
            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            
            return out
    
    model = MultiLayerPerceptron(input_size, hidden_size, num_classes).to(device)
    # Print model's state_dict
    '''
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    '''
    Key = len(hidden_size)-2
    History[Key] = [[],[]]
    
    # plt.plot(range(10),History[2][1])
    
    if train:
        model.apply(weights_init)
        model.train() #set dropout and batch normalization layers to training mode
    
        # Loss and optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=reg)
        criterion = nn.CrossEntropyLoss()
    
        # Train the model
        lr = learning_rate
        total_step = len(train_loader)
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(train_loader):
                # Move tensors to the configured device
                images = images.to(device)
                labels = labels.to(device)
                #################################################################################
                # TODO: Implement the training code                                             #
                # 1. Pass the images to the model                                               #
                # 2. Compute the loss using the output and the labels.                          #
                # 3. Compute gradients and update the model using the optimizer                 #
                # Use examples in https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
                #################################################################################
                # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
                
                with torch.set_grad_enabled(True):
                    output = model(images)
                    loss = criterion(output,labels)
                    loss.backward()
                    optimizer.step()
                optimizer.zero_grad()
                
                # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
                if (i+1) % 100 == 0:
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                           .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
    
            # Code to update the lr
            lr *= learning_rate_decay
            update_lr(optimizer, lr)
            with torch.no_grad():
                correct = 0
                total = 0
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    ####################################################
                    # TODO: Implement the evaluation code              #
                    # 1. Pass the images to the model                  #
                    # 2. Get the most confident predicted class        #
                    ####################################################
                    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
                    predicted = model(images)
                    predicted = torch.argmax(predicted,dim=1)
                    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                
                History[Key][0].append(loss.item())
                History[Key][1].append(100 * correct / total)                       
                print('Validataion accuracy is: {} %'.format(100 * correct / total))
    
        ##################################################################################
        # TODO: Now that you can train a simple two-layer MLP using above code, you can  #
        # easily experiment with adding more layers and different layer configurations   #
        # and let the pytorch library handle computing the gradients                     #
        #                                                                                #
        # Experiment with different number of layers (at least from 2 to 5 layers) and   #
        # record the final validation accuracies Report your observations on how adding  #
        # more layers to the MLP affects its behavior. Try to improve the model          #
        # configuration using the validation performance as the guidance. You can        #
        # experiment with different activation layers available in torch.nn, adding      #
        # dropout layers, if you are interested. Use the best model on the validation    #
        # set, to evaluate the performance on the test set once and report it            #
        ##################################################################################
    
        # Save the model checkpoint
        torch.save(model.state_dict(), 'model.ckpt')
         
    else:
        # Run the test code once you have your by setting train flag to false
        # and loading the best model
    
        best_model = None
        best_model = torch.load('model.ckpt')
        
        model.load_state_dict(best_model)
        
        # Test the model
        model.eval() #set dropout and batch normalization layers to evaluation mode
        
        # In test phase, we don't need to compute gradients (for memory efficiency)
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                ####################################################
                # TODO: Implement the evaluation code              #
                # 1. Pass the images to the model                  #
                # 2. Get the most confident predicted class        #
                ####################################################
                # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
                predicted = model(images)
                predicted = torch.argmax(predicted, 1)
                
                # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                if total == 1000:
                    break
    
            print('Accuracy of the network on the {} test images: {} %'.format(total, 100 * correct / total))
    



import matplotlib.pyplot as plt
figure, axis = plt.subplots(2, 1)
figure.legend(['label1', 'label2', 'label3'])

for KEY in History:
    LOSS, ACC = History[KEY]
    axis[0].plot(range(10),LOSS,label=str(KEY)+" Layers -- Train Loss: {:.4} -- Validation Acc: {:.4}% ".format(str(LOSS[-1]),str(ACC[-1])))
    axis[0].set_title("Loss")
      
    # For Cosine Function
    axis[1].plot(range(10),ACC)
    axis[1].set_title("Accuracy")
figure.legend()


