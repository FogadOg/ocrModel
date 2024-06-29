import cv2
import numpy as np
import torch  # Make sure to import torch for tensor operations

class Plot():
    def __init__(self, tokenizer, save_path: str = None) -> None:
        self.tokenizer = tokenizer
        self.save_path = save_path

    def renderRectangles(self, sample):
        image = sample["image"]
        for rect in sample["dataPoints"]:
            x = int(rect["x"])
            y = int(rect["y"])
            width = int(rect["width"])
            height = int(rect["height"])
            tag = rect["tag"]
            
            cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
            
            cv2.putText(image, tag, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if self.save_path:
            cv2.imwrite(self.save_path, image)
        else:
            cv2.imshow('Image', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    def renderPrediction(self, image_tensor, dataPoints, save_path=None):
        image = image_tensor.permute(1, 2, 0).numpy() * 255
        image = image.astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        for rect in dataPoints:
            x = int(rect[0])
            y = int(rect[1])
            width = int(rect[2])
            height = int(rect[3])

            word = self.tokenizer.decode(torch.tensor([rect[4]]))
            tag = str(word)
            
            cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
            
            cv2.putText(image, tag, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if save_path:
            cv2.imwrite(save_path, image)
        else:
            cv2.imshow('Prediction', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == "__main__":
    sample = {
        "image": np.zeros((500, 500, 3), dtype=np.uint8),
        "dataPoints": [
            {"x": 50, "y": 50, "width": 100, "height": 100, "tag": "Example1"},
            {"x": 200, "y": 200, "width": 150, "height": 150, "tag": "Example2"}
        ]
    }
    
    plot = Plot(save_path="output_image.png")
    plot.renderRectangles(sample)

    image_tensor = torch.rand(3, 500, 500)
    dataPoints = [
        [50, 50, 100, 100, "Example1"],
        [200, 200, 150, 150, "Example2"]
    ]
    plot.renderPrediction(image_tensor, dataPoints, save_path="prediction_image.png")
