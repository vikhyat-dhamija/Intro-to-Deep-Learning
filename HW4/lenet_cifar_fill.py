from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision 
import torchvision.transforms as transforms
import time

# Preparing for Data
print('==> Preparing data..')

# Training Data augmentation
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
# Testing Data preparation
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

#classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class LeNet(nn.Module):
    
    def __init__(self):
        
        super(LeNet, self).__init__()       
        #Sequential Model for the Convolution Neural Network
        self.convnet=nn.Sequential(nn.Conv2d(3, 6, 5),nn.ReLU(),nn.MaxPool2d(2,2),nn.Conv2d(6, 16, 5),nn.ReLU(),nn.MaxPool2d(2,2),nn.Conv2d(16, 120, 5),nn.ReLU())        
        #Sequential Model for the Fully Connected Layers ahead of the Convolution Part
        self.fc=nn.Sequential(nn.Linear(120,84),nn.ReLU(),nn.Linear(84,10),nn.LogSoftmax(dim=-1))
    

    def forward(self, x):       
        out = self.convnet(x)
        out = self.fc(out.view(-1,120))
        return out



def train(model, train_loader, optimizer, epoch):
    model.train()
    count = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        #data, target = data.to(device), target.to(device)       
        count+=1
        optimizer.zero_grad()#resetting the gradient       
        output=model(data)#data is passed to the model set so as to come out with output of forward propagation              
        loss_fn=nn.CrossEntropyLoss()#Using cross entropy loss
        loss=loss_fn(output,target)#calculate the loss based on the output and the target
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset),100. * batch_idx / len(train_loader), loss.item()))

def test( model,test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            #data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    time0 = time.time()
    # Training settings
    batch_size = 128
    epochs = 10
    lr = 0.05
    no_cuda = True
    save_model = False
    use_cuda = not no_cuda and torch.cuda.is_available()
    torch.manual_seed(100)
    #device = torch.device("cuda" if use_cuda else "cpu")
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

    model = LeNet()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    for epoch in range(1, epochs + 1):
        train( model, train_loader, optimizer, epoch)
        test( model, test_loader)

    if (save_model):
        torch.save(model.state_dict(),"cifar_lenet.pt")
    time1 = time.time() 
    print ('Traning and Testing total excution time is: %s seconds ' % (time1-time0))   


if __name__ == '__main__':
    main()
