"""
Demonstration script for leaf segmentation using YOLOv8.
"""

import os
import cv2
import matplotlib.pyplot as plt
from src.inference.segmentation import LeafSegmentation

def plot_results(image, results):
    """Plot original image, segmentation masks, and visualization."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot original image
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Plot aggregated mask
    axes[1].imshow(results['aggregated_mask'], cmap='jet')
    axes[1].set_title('Segmentation Masks')
    axes[1].axis('off')
    
    # Plot visualization
    axes[2].imshow(cv2.cvtColor(results['visualization'], cv2.COLOR_BGR2RGB))
    axes[2].set_title('Visualization')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    # Initialize segmentation model
    segmenter = LeafSegmentation()
    
    # Process sample images
    sample_dir = os.path.join(os.path.dirname(__file__), 'data')
    
    for img_file in os.listdir(sample_dir):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(sample_dir, img_file)
            print(f"\nProcessing {img_file}...")
            
            # Read image
            image = cv2.imread(img_path)
            
            # Perform segmentation
            results = segmenter.segment_image(image, visualize=True)
            
            # Show results
            print(f"Found {len(results['masks'])} leaves")
            plot_results(image, results)

if __name__ == "__main__":
    main()