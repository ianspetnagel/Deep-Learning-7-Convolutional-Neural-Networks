import Utils
import sys
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Resnet50Model import Bottleneck, ResNet, ResNet50

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = ResNet50(10).to(device)
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    _,trainloader,_,testloader = Utils.get_loaders()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.1, patience=5)
    EPOCHS = 200
    for epoch in range(EPOCHS):
        losses = []
        running_loss = 0
        for i, inp in enumerate(trainloader):
            inputs, labels = inp
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i%100 == 0 and i > 0:
                print(f'Loss [{epoch+1}, {i}](epoch, minibatch): ', running_loss / 100)
                running_loss = 0.0

        avg_loss = sum(losses)/len(losses)
        scheduler.step(avg_loss)

    print('---------------Training Done-------------')
    #----------------compute accuracy-----------------
    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to('cuda'), labels.to('cuda')
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy on 10,000 test images: ', 100*(correct/total), '%')

if __name__ == "__main__":
    sys.exit(int(main() or 0))