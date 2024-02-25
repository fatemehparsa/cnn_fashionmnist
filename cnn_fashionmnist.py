#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import time
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from torch.utils.data import random_split

params = {
        'net': 'cnn_fashionmnist',
        'device': "cuda:0" if torch.cuda.is_available() else "cpu",
        'num_epochs': 50,
        'milestones': [150, 225, 262],
        'activation_type': 'relu',
        'save_memory': False,
        'lipschitz': False,
        'lmbda': 1e-4,
        'lr': 1e-1,
        'aux_lr': 1e-3,
        'weight_decay': 5e-4,
        'valid_log_step': 5,  # 
        'dataset_name': 'fashionmnist',
        'batch_size': 64,
        'plot_imgs': False,
        'verbose': False,
        'knot_threshold': 0.,
        'num_classes': 10,
    }

class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 *14*14, 32)
        self.batch_norm4 = nn.BatchNorm1d(32)
        self.fc1_relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        self.fc2 = nn.Linear(32, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.batch_norm1(x)
        x = self.maxpool1(x)
               
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.batch_norm4(x)
        x = self.fc1_relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.softmax(x)

        return x

# Instantiate the model
net = net()
# Print the model summary
print(net)

def train_model(model, train_loader, optimizer, criterion, device,lipschitz):
    model.train()
    running_loss = 0.0
    all_predictions = []
    all_labels = []
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    train_loss = running_loss / len(train_loader)
    return  all_predictions, all_labels, train_loss


def test_model(model, test_loader, device,criterion):
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            loss = criterion(outputs, labels)
            running_loss += loss.item()
        test_loss = running_loss / len(test_loader)
    return  all_predictions, all_labels, test_loss


def plot_loss_accuracy(train_losses, validation_losses, train_accuracies, validation_accuracies):
    # Plot Training and Validation Loss
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(validation_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    # Plot Training and Validation Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(validation_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    ########################################################################
    # Load the data

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])

    batch_size = params['batch_size']

    trainset = torchvision.datasets.FashionMNIST(root='/kaggle/working'
                                                 ,train=True,
                                                 download=True, transform=transform)
    # Define the split ratios
    train_ratio = 0.9
    valid_ratio = 0.1

    # Calculate the sizes of training and validation sets
    train_size = int(train_ratio * len(trainset))
    valid_size = len(trainset) - train_size

    # Split the training set into training and validation sets
    trainset, validset = random_split(trainset, [train_size, valid_size])

    validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size,
                                              shuffle=False, num_workers=4)

    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=4)

    testset = torchvision.datasets.FashionMNIST(root='/kaggle/working',
                                                train=False,
                                                download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=4)
    
    for i in range(5):
                                             
        image, label = trainset[i]

        # Convert the PyTorch tensor to a NumPy array and transpose the dimensions
        image_np = image.numpy().transpose((1, 2, 0))

        # Display the image
        plt.imshow(image_np, cmap='gray')
        plt.title(f'Label: {label}')
        plt.show()
    
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'\nDevice: {device}')

    ########################################################################
    # Network, optimizer, loss

    net.to(device)
    print('DeepSpline: nb. parameters - {:d}'.format(sum(p.numel() for p in net.parameters())))
    # Wrap the model with DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)
    #net.load_state_dict(torch.load('/kaggle/input/model-resnet-fer2013-ds/model_resnetferds.pth'))
    optimizer = optim.SGD(net.module.parameters() if hasattr(net, 'module') else net.parameters(),
                               lr=0.001,
                               momentum=0.9)

    criterion = nn.CrossEntropyLoss()
    ########################################################################
 
    # Lists to store training loss and test accuracy for each epoch
    train_losses = []
    validation_losses = []
    train_accuracies = []
    validation_accuracies = []

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=params['milestones'], gamma=0.1)

    ########################################################################
    # Training the DeepSpline network
    # Regularization weight for the TV(2)/BV(2) regularization
    # Needs to be tuned for performance
    lmbda = 1e-4
    # lipschitz control: if True, BV(2) regularization is used instead of TV(2)
    lipschitz = False
    print('\nTraining relu network on fasionmnist.')
    start_time = time.time()

    for epoch in range(params['num_epochs']):
        train_predictions, train_labels,train_loss = train_model(net, trainloader, optimizer, criterion, device,lipschitz)
        train_losses.append(train_loss)
        train_accuracy = accuracy_score(train_labels, train_predictions)
        train_accuracies.append(train_accuracy)
        # Learning rate scheduling step
        scheduler.step()
        print(f'Epoch {epoch + 1}/{params["num_epochs"]}')
        print(f'train_acc :{train_accuracy* 100:.2f}%')
        print(f'train loss :{train_loss:.4f}')
        predictions, labels,validation_loss = test_model(net, validloader, device,criterion)
        acc_score = accuracy_score(labels, predictions)
        validation_accuracies.append(acc_score)
        validation_losses.append(validation_loss)

        if epoch % params['valid_log_step'] == 0:
            # Save the trained model
            model_filename = 'model_cnn_fashionmnist_relu.pth'
            torch.save(net.state_dict(), model_filename)

            print(f'validation_acc: {acc_score * 100:.2f}%')
            print(f'validation_loss: {validation_loss:.4f}')

            # Confusion Matrix
            cm = confusion_matrix(labels, predictions)
            print("Confusion Matrix:")
            print(cm)

            # Plot confusion matrix for validation
            plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
            plt.title("Confusion Matrix on validation set")
            plt.colorbar()
            plt.xticks(range(params['num_classes']), classes, rotation=45)
            plt.yticks(range(params['num_classes']), classes)
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.show()
            #plot train and validation
            plot_loss_accuracy(train_losses,validation_losses, train_accuracies,validation_accuracies)

    end_time = time.time()

    print('Finished Training relu network. \n'
          'Took {:d} seconds. '.format(int(end_time - start_time)))
    

    ######################################################################

    # Test the model on the test set after training
    test_predictions, test_labels ,test_loss= test_model(net, testloader, device,criterion)
    test_acc_score = accuracy_score(test_labels, test_predictions)
    print(f'Final Accuracy on test set: {test_acc_score * 100:.2f}%')
    print(f'Final Test Loss: {test_loss:.4f}')

    # Confusion Matrix for test set
    test_cm = confusion_matrix(test_labels, test_predictions)
    print("Confusion Matrix for Test Set:")
    print(test_cm)

    # Plot confusion matrix for test set
    plt.imshow(test_cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix for Test Set")
    plt.colorbar()
    plt.xticks(range(params['num_classes']), classes, rotation=45)
    plt.yticks(range(params['num_classes']), classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
    

    # Save the trained model
    model_filename = 'model_cnn_fashionmnist_relu.pth'
    torch.save(net.state_dict(), model_filename)


