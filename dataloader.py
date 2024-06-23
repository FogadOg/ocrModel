from torch.utils.data import DataLoader
import torch


class Dataloader():
  def __init__(self, dataset, trainSize: float):
    self.dataset=dataset
    self.trainSize=trainSize

  def trainTestDataloader(self):
    trainDataset, testDataset=self.splitDataset(self.dataset)

    trainDataloader=DataLoader(trainDataset, batch_size=64, shuffle=True)
    testDataloader=DataLoader(testDataset, batch_size=64, shuffle=True)
    return trainDataloader, testDataloader

  def splitDataset(self):
    trainSize = int(self.trainSize * len(self.dataset))
    testSize = len(self.dataset) - trainSize
    trainDataset, testDataset = torch.utils.data.random_split(self.dataset, [trainSize, testSize])

    return trainDataset, testDataset