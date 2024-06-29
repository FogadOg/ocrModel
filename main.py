from dataset import Dataset
import torch
from trainer import Trainer
from dataloader import Dataloader
from torch import nn
from model.model import Model

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

dataset = Dataset('SceneTrialTest/words.xml')
vocabSize = len(dataset.tokenizer)

model = Model(vocabSize, 30)

dataloader = Dataloader(dataset, .8)
trainDataloader, testDataloader = dataloader.splitDataset()


optimizer = torch.optim.Adam(model.parameters(), lr=.001)

# from torch.optim.lr_scheduler import ReduceLROnPlateau
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20, verbose=True)

criterionBbox = nn.SmoothL1Loss()


Trainer(trainDataloader, testDataloader, model, criterionBbox, optimizer, 6000, device, dataset.tokenizer)