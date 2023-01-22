import torch
import torch.nn as nn
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import time

# hyper-parameters
input_features = 784
num_classes = 10
num_epoch = 5
batch_size = 100
learning_rate = 0.002
l1_weight = 0.001
l2_weight = 0.001

# load dataset
train_data = torchvision.datasets.MNIST(root='/tmp/data', train=True, download=True, transform=transforms.ToTensor())
test_data = torchvision.datasets.MNIST(root='/tmp/data', train=False, transform=transforms.ToTensor())

# initiate data loader
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

class LogisticRegression(nn.Module):
    def __init__(self, input_features, num_classes, num_epoch, learning_rate, l1_weight=0, l2_weight=0):
        super(LogisticRegression, self).__init__()
        self.fc1 = nn.Linear(input_features, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.input_features = input_features
        self.num_epoch = num_epoch
        self.learning_rate = learning_rate
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
    
    def train(self, train_loader, model_name='logistic_regression.model', output_log_freq=0):
        """
        Train the model with given train_loader. Save the model if model name specified.
        """
        total = len(train_loader)
        for e in range(self.num_epoch):
            for i, (instances, labels) in enumerate(train_loader):
                instances = instances.reshape(-1, self.input_features).to(device)
                labels = labels.to(device)
                # Forward
                x = F.relu(self.fc1(instances))
                x = F.relu(self.fc2(x))
                output = F.relu(self.fc3(x))
                # Calculate loss
                params = torch.cat([x.view(-1) for x in self.parameters()])
                l1_loss = 0 if self.l1_weight == 0 else torch.norm(params, 1)
                l2_loss = 0 if self.l2_weight == 0 else torch.norm(params, 2)
                loss = self.criterion(output, labels) + self.l1_weight * l1_loss + self.l2_weight * l2_loss
                # Update weights
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if output_log_freq and (i + 1) % output_log_freq == 0:
                    print('Epoch %d/%d, trained %d/%d instances, Logloss: %.5f' % 
                          (e, self.num_epoch, i + 1, total, loss.item()))
        if model_name:
            torch.save(self.state_dict(), model_name)
            
    def predict(self, instances):
        """
        Predict the label with given training instance batch.
        """
        with torch.no_grad():
            instances = instances.reshape(-1, self.input_features)
            x = F.relu(self.fc1(instances))  # tensor with dim [batch_size, 10] 
            x = F.relu(self.fc2(x))
            output = F.relu(self.fc3(x))
            return torch.max(output.data, 1)[1]  # idx of the max element for each instance indicates the class

# Train
start_time = time.time()
print('Start training: %s' % start_time)
lr = LogisticRegression(input_features, num_classes, num_epoch, learning_rate, l1_weight, l2_weight).to(device)
lr.train(train_loader, output_log_freq=100)
print('End training: %s' % (time.time() - start_time))
# Evaluate
correct, total = 0, 0

for images, labels in test_loader:
    images = images.to(device)
    labels = labels.to(device)
    total += labels.size(0)
    predicted = lr.predict(images)
    correct += (predicted == labels).sum()

print('Accuracy: %d/%d' % (correct, total))