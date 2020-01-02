import torch
import torchvision
import numpy as np
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Pytorch():
    def __init__(self):
        self.net = Net()

    def train(self, X_train, y_train):
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.net.parameters(), lr=0.001)
        for epoch in range(1500):  # 5 full passes over the data
            X, y = torch.tensor(X_train), torch.tensor(y_train)  # X is the batch of features, y is the batch of targets.
            self.net.zero_grad()  # sets gradients to 0 before loss calc. You will do this likely every step.
            output = self.net(X.view(-1, len(X_train[0])))  # pass in the reshaped batch
            loss = loss_function(output, y)  # calc and grab the loss value
            loss.backward()  # apply this loss backwards thru the network's parameters
            optimizer.step()  # attempt to optimize weights to account for loss/gradients

    def test(self, X_test, y_test):
        predictions = []
        with torch.no_grad():
            X, y = torch.tensor(X_test), torch.tensor(y_test)
            output = self.net(X.view(-1, len(X_test[0])))
            for idx, i in enumerate(output):
                predictions.append(torch.argmax(i))
        return predictions

    def score(self, X_test, y_test):
        predictions = self.test(X_test, y_test)
        correct = 0
        for i in range(len(predictions)):
            if predictions[i] == y_test[i]:
                correct += 1
        return correct/len(predictions)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x.float()))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)