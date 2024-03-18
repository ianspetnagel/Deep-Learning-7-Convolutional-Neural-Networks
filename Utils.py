import torch
import torchvision
import torchvision.transforms as transforms

def get_loaders():
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])

    transform_test = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])
    trainds = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainds, batch_size=128, shuffle=True,num_workers=2)
    testds = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,transform=transform_test)
    testloader = torch.utils.data.DataLoader(testds, batch_size=128,shuffle=False,num_workers=2)
    return trainds, trainloader, testds, testloader
