import xml.etree.ElementTree as ET
import cv2
import torch

class Dataset():
    def __init__(self, filePath: str) -> None:
        self.dataset = []
        self.folderName = "SceneTrialTest/"
        self.filePath = filePath

        tree = ET.parse(self.filePath)
        root = tree.getroot()

        for image in root.findall('image'):
            sample = dict()
            dataPoints = []

            imageName = image.find('imageName').text
            sample["image"] = cv2.imread(self.folderName+imageName) 
            
            taggedRectangles = image.find('taggedRectangles')
            for rect in taggedRectangles.findall('taggedRectangle'):
                dataPoint = dict()
                dataPoint["x"] = rect.attrib['x']
                dataPoint["y"] = rect.attrib['y']
                dataPoint["width"] = rect.attrib['width']
                dataPoint["height"] = rect.attrib['height']
                dataPoint["offset"] = rect.attrib['offset']
                dataPoint["rotation"] = rect.attrib['rotation']
                dataPoint["userName"] = rect.attrib['userName']
                dataPoint["tag"] = rect.find('tag').text
                
                dataPoints.append(dataPoint)
            sample["dataPoints"] = dataPoints
            self.dataset.append(sample)
    
        

        
if __name__ == "__main__":
    dataset = Dataset('SceneTrialTest/words.xml')



