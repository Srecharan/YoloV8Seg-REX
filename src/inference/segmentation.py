"""
YOLOv8 Leaf Segmentation Module
Provides functionality for leaf detection and segmentation using YOLOv8.
"""

import os
from pathlib import Path
import numpy as np
import cv2
import torch
from ultralytics import YOLO
from typing import Union, Tuple, List, Dict

class LeafSegmentation:
    """A class for performing leaf segmentation using YOLOv8."""
    
    def __init__(self, 
                 model_path: str = None,
                 conf_threshold: float = 0.7,
                 iou_threshold: float = 0.2,
                 device: str = None):
        """
        Initialize the leaf segmentation model.
        
        Args:
            model_path (str, optional): Path to the YOLOv8 model weights.
                If None, will use the default model path from config.
            conf_threshold (float): Confidence threshold for detections.
            iou_threshold (float): IoU threshold for NMS.
            device (str, optional): Device to run the model on ('cuda' or 'cpu').
                If None, will automatically select available device.
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # Load model
        if model_path is None:
            # Default to the model in the package
            model_path = os.path.expanduser('~/ultralytics/large_best.pt')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
            
        self.model = YOLO(model_path)
        
        # Warm up the model
        self._warmup()
    
    def _warmup(self, size: Tuple[int, int] = (832, 1088)):
        """
        Warm up the model with a dummy inference.
        
        Args:
            size (tuple): Size of the dummy input (height, width).
        """
        dummy_input = np.zeros((*size, 3), dtype=np.uint8)
        self.model.predict(
            dummy_input,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device
        )
    
    def segment_image(self, 
                     image: Union[str, np.ndarray],
                     visualize: bool = False) -> Dict:
        """
        Perform leaf segmentation on an input image.
        
        Args:
            image: Either path to image file or numpy array of image
            visualize: Whether to return visualization of the segmentation
        
        Returns:
            dict: Contains following keys:
                - masks: List of binary masks for each detected leaf
                - scores: Confidence scores for each detection
                - visualization: BGR image with visualized results (if visualize=True)
        """
        # Load image if path is provided
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image not found at {image}")
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run inference
        results = self.model.predict(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            retina_masks=True
        )[0]
        
        # Process results
        if results.masks is None:
            return {
                'masks': [],
                'scores': [],
                'visualization': image if visualize else None
            }
        
        # Get masks
        masks = results.masks.data.cpu().numpy()
        
        # Create aggregated mask
        height, width = image.shape[:2]
        mask_aggregated = np.zeros((height, width))
        
        for idx, mask in enumerate(masks, 1):
            # Resize mask to original image size
            mask_resized = cv2.resize(
                mask.astype(float), 
                (width, height), 
                interpolation=cv2.INTER_NEAREST
            )
            mask_aggregated += mask_resized * idx
            
        # Prepare visualization if requested
        viz_image = None
        if visualize:
            viz_image = self._visualize_masks(image, masks, results.boxes.conf.cpu().numpy())
        
        return {
            'masks': masks,
            'scores': results.boxes.conf.cpu().numpy(),
            'aggregated_mask': mask_aggregated,
            'visualization': viz_image
        }
    
    def _visualize_masks(self, 
                        image: np.ndarray, 
                        masks: np.ndarray,
                        scores: np.ndarray) -> np.ndarray:
        """
        Create visualization of the segmentation results.
        
        Args:
            image: Original image
            masks: Binary masks from segmentation
            scores: Confidence scores
            
        Returns:
            np.ndarray: Visualization image
        """
        viz_image = image.copy()
        
        # Create random colors for each mask
        colors = np.random.randint(0, 255, size=(len(masks), 3))
        
        # Apply each mask
        for idx, (mask, score) in enumerate(zip(masks, scores)):
            color = colors[idx]
            
            # Resize mask to image size
            mask = cv2.resize(
                mask.astype(float),
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
            
            # Apply colored mask
            viz_image = np.where(
                mask[..., None],
                viz_image * 0.5 + color * 0.5,
                viz_image
            ).astype(np.uint8)
            
            # Add score text
            y_pos = 30 + idx * 30
            cv2.putText(
                viz_image,
                f"Leaf {idx+1}: {score:.2f}",
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color.tolist(),
                2
            )
        
        return viz_image