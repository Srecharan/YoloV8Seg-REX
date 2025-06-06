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
import matplotlib.pyplot as plt
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
                     visualize: bool = False,
                     output_json: bool = False,
                     output_path: str = None) -> Dict:
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
        
        masks = results.masks.data.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        
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
            viz_image = self._visualize_masks(image, masks, scores)
        
        # Calculate leaf statistics
        leaf_stats = self._calculate_leaf_stats(masks, height, width)
        
        # Export to JSON if requested
        if output_json:
            self._export_to_json(leaf_stats, scores, output_path)
        
        return {
            'masks': masks,
            'scores': scores,
            'aggregated_mask': mask_aggregated,
            'visualization': viz_image,
            'leaf_stats': leaf_stats
        }
    
    def _calculate_leaf_stats(self, masks: np.ndarray, height: int, width: int) -> List[Dict]:
        """
        Calculate statistics for each leaf mask.
        
        Args:
            masks: Binary masks for each leaf
            height: Image height
            width: Image width
            
        Returns:
            List of dictionaries containing stats for each leaf
        """
        leaf_stats = []
        
        for idx, mask in enumerate(masks):
            # Resize mask to original image size
            mask_resized = cv2.resize(
                mask.astype(float),
                (width, height),
                interpolation=cv2.INTER_NEAREST
            )
            
            # Convert to binary mask
            mask_binary = mask_resized.astype(np.uint8)
            
            # Find contours
            contours, _ = cv2.findContours(
                mask_binary,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if contours:
                # Get largest contour
                contour = max(contours, key=cv2.contourArea)
                
                # Calculate statistics
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                
                # Calculate centroid
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = 0, 0
                
                # Calculate bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Store stats
                leaf_stats.append({
                    'leaf_id': idx + 1,
                    'area_pixels': float(area),
                    'perimeter_pixels': float(perimeter),
                    'centroid': (int(cx), int(cy)),
                    'bounding_box': {
                        'x': int(x),
                        'y': int(y),
                        'width': int(w),
                        'height': int(h)
                    },
                    'aspect_ratio': float(w) / h if h != 0 else 0,
                    'circularity': 4 * np.pi * area / (perimeter * perimeter) if perimeter != 0 else 0
                })
            
        return leaf_stats

    def _export_to_json(self, leaf_stats: List[Dict], scores: np.ndarray, output_path: str = None):
        """
        Export leaf statistics to JSON file.
        
        Args:
            leaf_stats: List of leaf statistics
            scores: Confidence scores
            output_path: Path to save JSON file
        """
        import json
        from datetime import datetime
        
        # Add confidence scores to stats
        for idx, stat in enumerate(leaf_stats):
            stat['confidence_score'] = float(scores[idx])
        
        # Prepare export data
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'total_leaves': len(leaf_stats),
            'leaves': leaf_stats
        }
        
        # Save to JSON if output path is provided
        if output_path:
            json_path = output_path.replace('.png', '_stats.json')
            with open(json_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            print(f"Leaf statistics saved to: {json_path}")
        
        return export_data

    def _visualize_masks(self, 
                    image: np.ndarray, 
                    masks: np.ndarray,
                    scores: np.ndarray) -> np.ndarray:
        """
        Create enhanced visualization of the segmentation results.
        """
        viz_image = image.copy()
        height, width = image.shape[:2]
        
        # Create distinct colors for each mask
        num_masks = len(masks)
        hsv_colors = np.linspace(0, 1, num_masks)
        colors = []
        for h in hsv_colors:
            # Convert HSV to RGB for more distinct colors
            rgb = plt.cm.hsv(h)[:3]
            # Make colors more vibrant
            rgb = [int(c * 255) for c in rgb]
            colors.append(rgb)
        
        # Create transparent overlay
        overlay = np.zeros_like(image)
        label_positions = []
        
        # Apply each mask with enhanced visibility
        for idx, (mask, score, color) in enumerate(zip(masks, scores, colors)):
            # Resize mask to image size
            mask = cv2.resize(
                mask.astype(float),
                (width, height),
                interpolation=cv2.INTER_NEAREST
            )
            
            # Find contours for label placement
            mask_binary = mask.astype(np.uint8)
            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find centroid of largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    label_positions.append((cx, cy))
                else:
                    label_positions.append((10, 30 + idx * 30))
            else:
                label_positions.append((10, 30 + idx * 30))
            
            # Apply colored mask with higher opacity
            overlay = np.where(
                mask[..., None],
                color,
                overlay
            ).astype(np.uint8)
        
        # Blend original image with overlay
        alpha = 0.4  # Opacity of the overlay
        viz_image = cv2.addWeighted(viz_image, 1, overlay, alpha, 0)
        
        # Add labels and scores at centroid positions
        for idx, (pos_x, pos_y) in enumerate(label_positions):
            label = f"Leaf {idx+1}: {scores[idx]:.2f}"
            
            # Add white background for text
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(viz_image, 
                        (pos_x - 5, pos_y - text_h - 5),
                        (pos_x + text_w + 5, pos_y + 5),
                        (255, 255, 255),
                        -1)
            
            # Add text
            cv2.putText(viz_image,
                    label,
                    (pos_x, pos_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    colors[idx],
                    2)
        
        return viz_image