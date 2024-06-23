import cv2




class Plot():
    def __init__(self, sample) -> None:
        self.renderRectangles(sample)

    def renderRectangles(self, sample):
        image = sample["image"]
        for rect in sample["dataPoints"]:
            x = int(rect["x"])
            y = int(rect["y"])
            width = int(rect["width"])
            height = int(rect["height"])
            tag = rect["tag"]
            
            # Draw rectangle
            cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
            
            # Put the tag text
            cv2.putText(image, tag, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display the image
        cv2.imshow('Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

