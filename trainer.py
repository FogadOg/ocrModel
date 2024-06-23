from torch.utils.data import DataLoader
from torch import nn
import torch

class Trainer():
    def __init__(self, testDataloader: DataLoader, trainDataloader: DataLoader, model: nn.Module, criterionBbox, optimizer: torch.optim.Adam, epochs: int, device):
        self.testDataloader = testDataloader
        self.trainDataloader = trainDataloader
        self.model = model

        self.criterionBbox = criterionBbox

        self.optimizer = optimizer
        self.epochs = epochs
        self.device = device

        self.start()

    def start(self):
        for epoch in range(self.epochs):
            self.train()

    def train(self):
        for image, targetData in self.trainDataloader:
            self.optimizer.zero_grad()
            bboxOutput = self.model(image)

            targetDataPoints = self.extractDatapoints(targetData)
            print("targetDataPoints: ",targetDataPoints.shape)
            print("bboxOutput.shape: ", bboxOutput.shape)

            loss = self.getLoss(bboxOutput, targetDataPoints)
            loss.backward()

            self.optimizer.step()

            print("loss: ",loss.item())
    
    def extractDatapoints(self, dataPoints):
        extractedDatapoints = []

        for dataPoint in dataPoints:
            extractedDataPoint = []

            extractedDataPoint.append(dataPoint["x"])
            extractedDataPoint.append(dataPoint["y"])
            extractedDataPoint.append(dataPoint["width"])
            extractedDataPoint.append(dataPoint["height"])
            extractedDataPoint.append(dataPoint["token"])

            extractedDatapoints.append(extractedDataPoint)
        return torch.tensor(extractedDatapoints)


    def getLoss(self, predDataPoints, targetDataPoints):
        pass

        # bboxLoss = self.criterionBbox(predDataPoints, targetDataPoints)

        # return bboxLoss
  