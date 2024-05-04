import dataProcessing as dp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm 


batchSize = 128
epochs = 15
classNum  = 25
class MalConvNet(nn.Module):
    def __init__(self, num_classes=25):
        super(MalConvNet, self).__init__()
        # Reduce the number of filters in each convolutional layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)  # Reduced from 50 to 32
        self.conv2 = nn.Conv2d(32, 48, kernel_size=3, padding=1)  # Reduced from 70 to 48
        self.conv3 = nn.Conv2d(48, 48, kernel_size=3, padding=1)  # Same as conv2

        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(48)
        self.bn3 = nn.BatchNorm2d(48)

        # Increase pooling stride to reduce spatial dimensions more significantly
        self.pool = nn.MaxPool2d(2, stride=2)  # Increased stride to 2

        # Reduce the size of the fully connected layers
        self.fc = nn.Linear(49152, 128)  # Adjusted for reduced feature map size
        self.fc_output = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        x = self.fc_output(x)
        return x


if torch.cuda.is_available():
    print("CUDA is available. Training on GPU.")
else:
    print("CUDA is not available. Training on CPU.")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load data
img, label, classNum = dp.readData()

# Encode label
labelEncoder = LabelEncoder()
label = labelEncoder.fit_transform(label)

# Convert to tensor
img = torch.tensor(img, dtype=torch.float32).permute(0, 3, 1, 2)
label = torch.tensor(label, dtype=torch.long)

# Create dataSet
dataSet = TensorDataset(img, label)
trainSize = int(0.8 * len(dataSet))
testSize = len(dataSet) - trainSize
trainDataSet, testDataSet = data.random_split(dataSet, [trainSize, testSize])
trainLoader = DataLoader(trainDataSet, batch_size=batchSize, shuffle=True)
testLoader = DataLoader(testDataSet, batch_size=batchSize, shuffle=False)

# Model setup
model = MalConvNet(classNum).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
model.train()
for i in range(epochs):
    allLabels = []
    allPreds = []
    totalLoss = 0

    progressBar = tqdm(trainLoader, desc=f'Epoch {i+1}/{epochs}', leave=True)

    for images, labels in progressBar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        totalLoss += loss.item()
        
        _, preds = torch.max(outputs, 1)
        allLabels.extend(labels.cpu().numpy())
        allPreds.extend(preds.cpu().numpy())
        # Update progress bar with loss information
        progressBar.set_postfix({'loss': f'{loss.item():.4f}'})

    averageLoss = totalLoss / len(trainLoader)
    accuracy = accuracy_score(allLabels, allPreds)
    precision = precision_score(allLabels, allPreds, average='macro')
    recall = recall_score(allLabels, allPreds, average='macro')
    f1 = f1_score(allLabels, allPreds, average='macro')

    print(f'Epoch {i+1}: Avg Loss = {averageLoss:.4f}, Accuracy = {accuracy:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}, F1 Score = {f1:.4f}')

# Evaluation
model.eval()
allLabels = []
allPreds = []
with torch.no_grad():
    for images, labels in testLoader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        allLabels.extend(labels.cpu().numpy())
        allPreds.extend(predicted.cpu().numpy())

accuracy = accuracy_score(allLabels, allPreds)
precision = precision_score(allLabels, allPreds, average='macro')
recall = recall_score(allLabels, allPreds, average='macro')
f1 = f1_score(allLabels, allPreds, average='macro')

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
