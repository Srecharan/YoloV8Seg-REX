"""
Demonstration script for leaf segmentation using YOLOv8.
"""

import os
import sys
import cv2
import matplotlib.pyplot as plt
from src.inference.segmentation import LeafSegmentation

def plot_results(image, results, save_path=None):
    """Plot original image, segmentation masks, and visualization."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot original image
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Plot aggregated mask
    if results['aggregated_mask'] is not None:
        mask_img = axes[1].imshow(results['aggregated_mask'], cmap='jet')
        plt.colorbar(mask_img, ax=axes[1], label='Leaf ID')
    axes[1].set_title('Segmentation Masks')
    axes[1].axis('off')
    
    # Plot visualization
    if results['visualization'] is not None:
        axes[2].imshow(cv2.cvtColor(results['visualization'], cv2.COLOR_BGR2RGB))
    axes[2].set_title('Visualization')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Results saved to: {save_path}")
    else:
        plt.show()

def process_image(segmenter, img_path, output_dir=None):
    """Process a single image and display/save results."""
    try:
        print(f"\nProcessing {os.path.basename(img_path)}...")
        
        # Read image
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to read image: {img_path}")
            
        # Perform segmentation
        results = segmenter.segment_image(image, visualize=True)
        
        # Create output directory if needed
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(img_path))[0]}_results.png")
        else:
            save_path = None
            
        # Show results
        print(f"Found {len(results['masks'])} leaves")
        if len(results['masks']) > 0:
            print(f"Confidence scores: {[f'{score:.2f}' for score in results['scores']]}")
        
        plot_results(image, results, save_path)
        
        return True
        
    except Exception as e:
        print(f"Error processing {img_path}: {str(e)}", file=sys.stderr)
        return False

def main():
    try:
        # Initialize segmentation model
        print("Initializing leaf segmentation model...")
        segmenter = LeafSegmentation()
        
        # Process sample images
        sample_dir = os.path.join(os.path.dirname(__file__), 'data')
        output_dir = os.path.join(os.path.dirname(__file__), 'results')
        
        if not os.path.exists(sample_dir):
            raise FileNotFoundError(f"Sample directory not found: {sample_dir}")
        
        # Process all images in the data directory
        success = 0
        total = 0
        
        for img_file in os.listdir(sample_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                total += 1
                img_path = os.path.join(sample_dir, img_file)
                if process_image(segmenter, img_path, output_dir):
                    success += 1
        
        print(f"\nProcessing complete! Successfully processed {success}/{total} images.")
        if output_dir:
            print(f"Results saved in: {output_dir}")
            
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()