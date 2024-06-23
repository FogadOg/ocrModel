from torch import nn
import torch
import torch.nn.functional as F



class Model(nn.Module):
    def __init__(self, numClasses, maxBoxes = 20):
        super().__init__()
        self.numClasses = numClasses
        self.maxBoxes = maxBoxes

        self.numDataPoints = 5

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 56 * 56, 512)
        
        self.fcBbox = nn.Linear(512, self.maxBoxes * self.numDataPoints)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)
        x = F.relu(self.fc1(x))
        
        
        bboxOutput = self.fcBbox(x).view(-1, self.maxBoxes, self.numDataPoints)
        
        return classOutput, bboxOutput





if __name__ == "__main__":
    model = Model(numClasses=300)

    # Create a sample input tensor
    sample_input = torch.randn(1, 3, 224, 224)

    # Pass the sample input through the model
    classOutput, bboxOutput = model(sample_input)

    # Print the shapes of the outputs
    print("Class Output Shape:", classOutput.shape)  # Expected: (1, 20, 300)
    print("Bounding Box Output Shape:", bboxOutput.shape)  # Expected: (1, 20, 4)




