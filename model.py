import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# The VGG11 net model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3,
                               stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,
                               stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3,
                               stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,
                               stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3,
                               stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                               stride=1, padding=1)
        self.conv7 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                               stride=1, padding=1)
        self.conv8 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                               stride=1, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        self.batchnorm1 = nn.BatchNorm2d(64)
        self.batchnorm2 = nn.BatchNorm2d(128)
        self.batchnorm3 = nn.BatchNorm2d(256)
        self.batchnorm4 = nn.BatchNorm2d(256)
        self.batchnorm5 = nn.BatchNorm2d(512)
        self.batchnorm6 = nn.BatchNorm2d(512)
        self.batchnorm7 = nn.BatchNorm2d(512)
        self.batchnorm8 = nn.BatchNorm2d(512)

        self.fc1 = nn.Linear(512, 4096)  # 6*6 from image dimension
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

    def forward(self, x):
        x = self.pool(F.relu(self.batchnorm1(self.conv1(x))))
        x = self.pool(F.relu(self.batchnorm2(self.conv2(x))))
        x = F.relu(self.batchnorm3(self.conv3(x)))
        x = self.pool(F.relu(self.batchnorm4(self.conv4(x))))
        x = F.relu(self.batchnorm5(self.conv5(x)))
        x = self.pool(F.relu(self.batchnorm6(self.conv6(x))))
        x = F.relu(self.batchnorm7(self.conv7(x)))
        x = self.pool(F.relu(self.batchnorm8(self.conv8(x))))
        x = x.view(-1, 512)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

    def train_test(self, trainloader, testloader, batch_size, device, num_epochs = 5):
        train_len = len(trainloader) * batch_size
        test_len = len(testloader) * batch_size

        # Record loss
        train_loss = []
        test_loss = []

        # Record accuracy
        train_accuracy = []
        test_accuracy = []

        for epoch in range(num_epochs):

            running_loss = 0.0

            for i, data in enumerate(trainloader, 0):

                # get the inputs
                inputs, labels = data

                # move to device
                inputs, labels = inputs.to(device), labels.to(device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self(inputs).to(device)
                loss = self.criterion(outputs, labels)

                loss.backward()
                self.optimizer.step()

                # store / print statistics
                running_loss += loss.item()

                if i % 5000 == 4999:  # print every 5000 mini-batches
                    print('[%d %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 5000))
                    running_loss = 0.0

            # Get the training loss & accuracy after each epoch
            correct = 0
            loss = 0
            with torch.no_grad():
                for data in trainloader:
                    # accuracy
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)
                    outputs = self(images).to(device)
                    _, predicted = torch.max(outputs.data, 1)
                    correct += (predicted == labels).sum().item()

                    # loss
                    loss += self.criterion(outputs, labels).item()

            train_loss.append(loss / train_len)
            train_accuracy.append(correct / train_len)

            # Get the test loss & accuracy after each epoch
            correct = 0
            loss = 0
            with torch.no_grad():
                for data in testloader:
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)
                    outputs = self(images).to(device)
                    _, predicted = torch.max(outputs.data, 1)
                    correct += (predicted == labels).sum().item()

                    # loss
                    loss += self.criterion(outputs, labels).item()

            test_accuracy.append(correct / test_len)
            test_loss.append(loss / test_len)

        print('Finished Training')
        return train_loss, train_accuracy, test_loss, test_accuracy
