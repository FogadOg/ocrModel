from dataset import Dataset
import torch
from trainer import Trainer
from dataloader import Dataloader
from torch import nn
from model.model import Model
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau


class Main():
    def __init__(self, 
                EPOCHS: int, 
                patience: int = None,
                saveModelPath: str = None,
                loadModelPath: str = None
            ):
        self.device = self.setUpDevice()

        self.dataset = Dataset('SceneTrialTest/words.xml')

        self.model = self.getModel()

        self.trainDataloader, self.testDataloader =  self.splitDataset()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=.0001)

        if patience != None:
            ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=patience, verbose=True)

        self.criterionBbox = nn.SmoothL1Loss()

        
        Trainer(
            self.testDataloader, 
            self.trainDataloader, 
            self.model, 
            self.criterionBbox, 
            self.optimizer, 
            EPOCHS, 
            self.device, 
            self.dataset.tokenizer,
            saveModelPath, 
            loadModelPath
        )
        

    def setUpDevice(self):
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(device)
        return device

    def getModel(self):
        vocabSize = len(self.dataset.tokenizer)
        model = Model(vocabSize, 30)

        return model.to(self.device)

    def splitDataset(self):        
        dataloader = Dataloader(self.dataset, .8)
        return dataloader.splitDataset()


Main(1000, saveModelPath = "model")