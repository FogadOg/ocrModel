import json

class TokenizerDatasetUtils():
  def __init__(self, maxDictLen=100000):
    self.maxDictLen=maxDictLen
    self.tokinizerDict=dict({"PAD":0, "UNK":1})

  def loadTokinizerDictionary(self, filePath)->None:
    jsonfile=open(filePath)
    jsonObject=json.load(jsonfile)
    self.tokinizerDict=jsonObject

  def createDataset(self, dataset: list[list[str]], fileSavePath=None)->None:
    for sentance in dataset:
      for word in sentance:
        if self.maxDictLen==len(self.tokinizerDict):
          break

        if word not in self.tokinizerDict:
          self.tokinizerDict[word]=len(self.tokinizerDict)

    if fileSavePath !=None:
      self._saveDictionary(fileSavePath)

    self.createDataset(dataset, fileSavePath)

  def _saveDictionary(self, savePath):
    with open(savePath, "w") as outfile:
      json.dump(self.tokinizerDict, outfile)