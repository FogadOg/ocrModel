import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, numClasses, hiddenSize, maxBoxes=20):
        super(Model, self).__init__()
        self.numClasses = numClasses
        self.maxBoxes = maxBoxes
        self.sequential = nn.Sequential(
            nn.Conv2d(3, hiddenSize, 3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),

            nn.Conv2d(hiddenSize, hiddenSize, 3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),

            nn.Conv2d(hiddenSize, hiddenSize, 3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),

            nn.Conv2d(hiddenSize, hiddenSize, 3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),

            nn.Conv2d(hiddenSize, hiddenSize, 3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),

            nn.Conv2d(hiddenSize, 32, 3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
        )
        self.fcClass = nn.Sequential(
            nn.Linear(2240, hiddenSize),
            nn.ReLU(),
            nn.Linear(hiddenSize, hiddenSize),
            nn.ReLU(),
            nn.Linear(hiddenSize, hiddenSize),
            nn.ReLU(),
            nn.Linear(hiddenSize, self.maxBoxes * numClasses)
        )

        self.fcBbox = nn.Linear(2240, self.maxBoxes * 4)

    def forward(self, x: torch.tensor):
        x = x.unsqueeze(0)
        x = self.sequential(x)
        x = x.view(x.size(0), -1)

        bboxOutput = self.fcBbox(x).view(-1, self.maxBoxes, 4)
        classOutput = self.fcClass(x).view(-1, self.maxBoxes, self.numClasses)

        classIndices = torch.argmax(classOutput, dim=2, keepdim=True).float()

        output = torch.cat((bboxOutput, classIndices), dim=2)

        return output.squeeze(0)

if __name__ == "__main__":
    model = Model(numClasses=769)

    sample_input = torch.randn(3, 640, 480)

    sample_input = sample_input.unsqueeze(0)

    output = model(sample_input)

    print(output)
