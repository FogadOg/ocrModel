from torch.utils.data import DataLoader
from torch import nn
import torch
from plot import Plot
class Trainer():
    def __init__(self, testDataloader: DataLoader, trainDataloader: DataLoader, model: nn.Module, criterionBbox, optimizer: torch.optim.Adam, epochs: int, device, tokienizer):
        self.testDataloader = testDataloader
        self.trainDataloader = trainDataloader
        self.model = model

        self.criterionBbox = criterionBbox

        self.optimizer = optimizer
        self.epochs = epochs
        self.currentEpoch = 0
        self.device = device

        self.tokienizer = tokienizer

        self.start()

    def start(self):
        for epoch in range(self.epochs):
            self.train()
            self.currentEpoch = epoch

    def train(self):
        for image, targetData in self.trainDataloader:
            self.optimizer.zero_grad()
            predBboxOutput = self.model(image)

            targetBboxOutput = self.extractDatapoints(targetData)

            loss, selectedPreds = self.getLoss(predBboxOutput, targetBboxOutput)
            


            loss.backward()

            self.optimizer.step()

        print("loss: ",loss.item())
        save_path = f"images/epoch_{self.currentEpoch}_batch_{self.currentEpoch}.png"
        plot = Plot(self.tokienizer, save_path)
        plot.renderPrediction(image.cpu(), selectedPreds.cpu().tolist(), save_path)
    
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


    def getLoss(self, predBboxOutput, targetBboxOutput):
        numBbox, _ = targetBboxOutput.shape
        assert predBboxOutput.shape[0] >= numBbox, f"Not enough predictions to select {numBbox} bounding boxes."

        selectedPreds = predBboxOutput[:numBbox]

        bboxLoss = self.criterionBbox(selectedPreds, targetBboxOutput)

        return bboxLoss, selectedPreds
  