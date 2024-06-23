from torch import nn
import torch
import torch.nn.functional as F



class Model(nn.Module):
    def __init__(self, numClasses = 300, maxBoxes = 20):
        super(self).__init__()
        self.numClasses = numClasses
        self.maxBoxes = maxBoxes

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 56 * 56, 512)
        
        # Linear layers for class scores and bounding box coordinates
        self.fcClass = nn.Linear(512, self.maxBoxes * self.numClasses)
        self.fcBbox = nn.Linear(512, self.maxBoxes * 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)
        x = F.relu(self.fc1(x))
        
        # Compute class scores and reshape to (batch_size, maxBoxes, numClasses)
        classOutput = self.fcClass(x).view(-1, self.maxBoxes, self.numClasses)
        
        # Compute bounding box coordinates and reshape to (batch_size, maxBoxes, 4)
        bboxOutput = self.fcBbox(x).view(-1, self.maxBoxes, 4)
        
        return classOutput, bboxOutput









