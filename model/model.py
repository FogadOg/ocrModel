import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, numClasses, maxBoxes=20):
        super(Model, self).__init__()
        self.numClasses = numClasses
        self.maxBoxes = maxBoxes
        self.sequential = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),

            nn.Conv2d(16, 16, 3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),

            nn.Conv2d(16, 16, 3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
        )

        
        self.fcBbox = nn.Linear(4800, self.maxBoxes * 5)

    def forward(self, x):
        x = self.sequential(x)
        
        x = x.view(x.size(0), -1)

        bboxOutput = self.fcBbox(x).view(-1, self.maxBoxes, 5)

        return bboxOutput







if __name__ == "__main__":
    model = Model(numClasses=300)

    # Create a sample input tensor
    sample_input = torch.randn(1, 3, 224, 224)

    # Pass the sample input through the model
    classOutput, bboxOutput = model(sample_input)

    # Print the shapes of the outputs
    print("Class Output Shape:", classOutput.shape)  # Expected: (1, 20, 300)
    print("Bounding Box Output Shape:", bboxOutput.shape)  # Expected: (1, 20, 4)




