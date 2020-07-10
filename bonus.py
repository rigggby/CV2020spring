import numpy as np
from PIL import Image
import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import matplotlib.pyplot as plt
from torchvision.transforms import Resize

def load_dataset(batch_size, train = True):
    if train == True:
        data_path = './train/'
    else:
        data_path = './test/'
    train_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=transforms.Compose([transforms.Resize([256, 256]), transforms.ToTensor()]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=1,
        shuffle=True
    )
    return train_loader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
num_classes = 15
model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model = model.to(device)
batch_size = 64
n_epoch = 10
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay = 5e-4)
criterion = nn.CrossEntropyLoss()
train_loader = load_dataset(batch_size)
test_loader = load_dataset(batch_size, False)

def train(model, train_loader, test_loader, n_epoch, optimizer, criterion, device, save_path = None):
    print('start train')
    highest_acc = 0
    train_acc_history = []
    test_acc_history = []
    for epoch in range(n_epoch):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader):           
            #print(i)
            inputs, targets = data[0].to(device, dtype=torch.float), data[1].to(device, dtype=torch.long)           
            #print(inputs.shape)
            optimizer.zero_grad()
            outputs = model.forward(inputs).to(device, dtype=torch.float)
            loss = criterion(outputs, targets)
            #print(outputs.shape)
            #print(targets.shape)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i%100 == 99:
                print(running_loss/100)
                running_loss = 0.0
        train_acc = test_accuracy(model, train_loader, device)
        test_acc = test_accuracy(model, test_loader, device)
        print(train_acc, test_acc)
        train_acc_history.append(train_acc)
        test_acc_history.append(test_acc)
        running_loss = 0.0
    print('test acc: ', highest_acc)
    print('FINISH!!')
    
    return train_acc_history, test_acc_history

def test_accuracy(model, data_loader, device):
    model.eval()
    test_accuracy = []
    total = 0
    correct = 0
    for i, data in enumerate(data_loader):
        x, y = data[0].to(device, dtype=torch.float), data[1].to(device, dtype=torch.float).view(-1)
        y_hat = model.forward(x)
        y_hat = y_hat.argmax(1) 
        for y_hat_, y_ in zip(y_hat, y):
            if y_hat_.long() == y_.long() :
                correct +=1
            total +=1

    return correct/total*100

train_acc, test_acc = train(model, train_loader, test_loader, n_epoch, optimizer, criterion, device, save_path = None)
