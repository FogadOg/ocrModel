from torch.utils.data import DataLoader
from torch import nn
import torch
from tqdm import tqdm
from plot import Plot
class Trainer():
    def __init__(self, 
                 testDataloader: DataLoader, 
                 trainDataloader: DataLoader, 
                 model: nn.Module, 
                 criterionBbox, 
                 optimizer: torch.optim.Adam, 
                 epochs: int, 
                 device, 
                 tokienizer,                 
                saveModelPath: str = None,
                loadModelPath: str = None
        ):
        self.testDataloader = testDataloader
        self.trainDataloader = trainDataloader
        self.model = model

        self.criterionBbox = criterionBbox

        self.optimizer = optimizer
        self.epochs = epochs
        self.currentEpoch = 0
        self.device = device

        self.tokienizer = tokienizer

        self.saveModelPath = saveModelPath
        self.saveModelPath = saveModelPath

        print(f"model parameters: {self.parametersCount(model):,}")
        if loadModelPath != None:
            self.loadModel(model, loadModelPath)
        self.start()

    def start(self):
        for epoch in tqdm(range(self.epochs)):
            self.train()
            self.currentEpoch = epoch


    def train(self):
        for image, targetData in self.trainDataloader:
            image = image.to(self.device)
            
            # Convert targetData to a tensor
            targetDataTensor = self.extractDatapoints(targetData).to(self.device)

            self.optimizer.zero_grad()
            predBboxOutput = self.model(image)

            loss, selectedPreds = self.getLoss(predBboxOutput, targetDataTensor)

            loss.backward()
            self.optimizer.step()

        if self.saveModelPath is not None:
            self.saveModel()
        print("loss: ", loss.item())

        if self.currentEpoch % 10 == 0:
            save_path = f"images/epoch_{self.currentEpoch}_batch_{self.currentEpoch}.png"
            plot = Plot(self.tokienizer, save_path)
            plot.renderPrediction(image.cpu(), selectedPreds.cpu().tolist(), save_path)



    def extractDatapoints(self, dataPoints):
        extractedDatapoints = []

        for dataPoint in dataPoints:
            extractedDataPoint = []

            # Ensure each value is converted to the appropriate numeric type
            extractedDataPoint.append(float(dataPoint["x"]))
            extractedDataPoint.append(float(dataPoint["y"]))
            extractedDataPoint.append(float(dataPoint["width"]))
            extractedDataPoint.append(float(dataPoint["height"]))
            extractedDataPoint.append(float(dataPoint["token"]))  # Assuming token is also numeric

            extractedDatapoints.append(extractedDataPoint)
            
        return torch.tensor(extractedDatapoints)

    def getGradient(self):
        print("Gradients:")
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                print(f"{name} gradient:\n{param.grad.shape}")


    def getLoss(self, predBboxOutput, targetBboxOutput):
        numBbox, _ = targetBboxOutput.shape
        assert predBboxOutput.shape[0] >= numBbox, f"Not enough predictions to select {numBbox} bounding boxes."

        selectedPreds = predBboxOutput[:numBbox]

        bboxLoss = self.criterionBbox(selectedPreds, targetBboxOutput)

        return bboxLoss, selectedPreds
    
    def saveModel(self):
        torch.save(self.model.state_dict(), f"{self.saveModelPath}.pth")
  
    def loadModel(self, model, loadModel):
        try:
            model.load_state_dict(torch.load(f"{loadModel}.pth"))
        except Exception as e:
            print("ERROR: ",e)

    def parametersCount(self, model: torch.nn.Module):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)