import xml.etree.ElementTree as ET
import cv2
import torch
from tokenizer import Tokenizer

class Dataset():
    def __init__(self, filePath: str, targetSize=(640, 480)) -> None:
        self.dataset = []
        self.tokenizer = Tokenizer(maxSquenceLength=1)
        self.folderName = "SceneTrialTest/"
        self.filePath = filePath
        self.targetSize = targetSize

        tree = ET.parse(self.filePath)
        root = tree.getroot()

        for image in root.findall('image'):
            sample = dict()
            dataPoints = []

            imageName = image.find('imageName').text
            resolution = image.find('resolution')
            originalWidth = int(resolution.attrib['x'])
            originalHeight = int(resolution.attrib['y'])
            sample["image"], scale_x, scale_y = self.normalize(self.folderName + imageName, originalWidth, originalHeight)

            sample["resolutionX"] = self.targetSize[0]
            sample["resolutionY"] = self.targetSize[1]

            taggedRectangles = image.find('taggedRectangles')
            for rect in taggedRectangles.findall('taggedRectangle'):
                dataPoint = dict()
                dataPoint["x"] = float(rect.attrib['x']) * scale_x
                dataPoint["y"] = float(rect.attrib['y']) * scale_y
                dataPoint["width"] = float(rect.attrib['width']) * scale_x
                dataPoint["height"] = float(rect.attrib['height']) * scale_y
                dataPoint["offset"] = float(rect.attrib['offset'])
                dataPoint["rotation"] = float(rect.attrib['rotation'])
                dataPoint["userName"] = rect.attrib['userName']
                dataPoint["tag"] = rect.find('tag').text

                token = self.tokenizer.encode(rect.find('tag').text)[0]
                dataPoint["token"] = token
                
                dataPoints.append(dataPoint)
                
            sample["dataPoints"] = dataPoints
            self.dataset.append(sample)
    
    def normalize(self, filePath: str, originalWidth: int, originalHeight: int):        
        cvImage = cv2.imread(filePath)
        cvImage = cv2.cvtColor(cvImage, cv2.COLOR_BGR2RGB)
        cvImage = cv2.resize(cvImage, self.targetSize)
        scale_x = self.targetSize[0] / originalWidth
        scale_y = self.targetSize[1] / originalHeight
        cvImage = torch.tensor(cvImage, dtype=torch.float32).permute(2, 0, 1) / 255.0

        return cvImage, scale_x, scale_y
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]["image"], self.dataset[idx]["dataPoints"]

if __name__ == "__main__":
    from plot import Plot
    plot = Plot()
    dataset = Dataset('SceneTrialTest/words.xml')
    plot.renderRectangles(dataset.dataset[0])
