from src.inference.segmentation import LeafSegmentation
import cv2
import numpy as np

def main():
    # Initialize segmenter
    segmenter = LeafSegmentation()
    
    # Load an image
    image_path = "path/to/your/image.jpg"
    image = cv2.imread(image_path)
    
    # Get segmentation masks
    masks = segmenter.process_image(image)
    
    # Visualize results
    segmenter.visualize(masks)

if __name__ == "__main__":
    main()
