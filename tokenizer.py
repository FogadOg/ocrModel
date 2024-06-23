import  sys, os
currentDir = os.path.dirname(os.path.abspath(__file__))
parentDir = os.path.abspath(os.path.join(currentDir, '../../'))
sys.path.append(parentDir)

import torch
from tokenizerUtil import TokenizerDatasetUtils

class Tokenizer(TokenizerDatasetUtils):
  def __init__(self, maxDictLen=100000, maxSquenceLength=700):
    super().__init__(maxDictLen)
    self.maxSquenceLength=maxSquenceLength

  def encode(self, sentance: str)->list[int]:
    encodedSentance=[]

    sentanceList = sentance.split(" ")
    for word in sentanceList:
      if len(encodedSentance) > self.maxSquenceLength:
        break

      if word in self.tokinizerDict.keys():
        encodedSentance.append(self.tokinizerDict[word])

      elif self.maxDictLen > len(self.tokinizerDict):
        self.tokinizerDict[word] = len(self.tokinizerDict)
        encodedSentance.append(self.tokinizerDict[word])
        
      else:
        encodedSentance.append(self.tokinizerDict["UNK"])


    return self.addPaddingToEncoding(encodedSentance)

  def addPaddingToEncoding(self, encoding: list[int])->list[int]:
    paddingLengthRequired=self.maxSquenceLength-len(encoding)
    paddingArray=[self.tokinizerDict["PAD"]]*paddingLengthRequired

    return encoding+paddingArray

  def decode(self, encodedSentance: torch.tensor)->list[str]:
    decodedString=""
    dictKeys=list(self.tokinizerDict.keys())

    for token in encodedSentance.cpu().detach().numpy():
      try:
        wordPosition=list(self.tokinizerDict.values()).index(token)
        decodedString+=dictKeys[wordPosition]+" "
      except ValueError:
        raise ValueError(f"token {token} was not found in dictionary")

    return decodedString

  def __len__(self):
    return len(self.tokinizerDict)

  def __getitem__(self, word)->int:
    return self.tokinizerDict[word]