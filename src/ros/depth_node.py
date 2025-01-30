#!/usr/bin/env python3
import sys
import os
from ultralytics import YOLO
import rospy
import numpy as np
from PIL import Image as PIL_image
from matplotlib import pyplot as plt
from sensor_msgs.msg import Image, CameraInfo
from message_filters import ApproximateTimeSynchronizer, Subscriber
import time
import torch
import gc
import cv2 as cv
from yoloV8_seg.msg import masks
from datetime import datetime
from cv_bridge import CvBridge

HOME_DIR = os.path.expanduser('~')

class YoloXDepthSeg:
    def __init__(self):
        self.bridge = CvBridge()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = os.path.join(HOME_DIR, 'SDF_OUT', timestamp)
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize parameters
        self.setup_parameters()
        
        # Setup subscribers and publishers
        self.setup_ros_interface()
        
        # Initialize model
        self.model = self.init_model()
        
        # Warm up GPU
        self.warmup_gpu()
        
        print('YOLOv8-X model initialization complete... waiting for images...')

    def setup_parameters(self):
        """Initialize all parameters and configurations"""
        self.receptive_width = 1088
        self.receptive_height = 832
        self.im_width = 1440
        self.im_height = 1080
        self.im_channels = 3
        self.count = 0

        # Model parameters
        self.conf = rospy.get_param('~conf_threshold', 0.7)
        self.iou = rospy.get_param('~iou_threshold', 0.2)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model configuration
        self.model_config = {
            'conf': self.conf,
            'iou': self.iou,
            'retina_masks': True,
            'hide_conf': True,
            'hide_labels': True,
            'boxes': False,
            'device': self.device
        }

    def setup_ros_interface(self):
        """Setup all ROS subscribers and publishers"""
        # Image and depth subscribers
        self.image_sub = Subscriber('/theia/left/image_rect_color', Image)
        self.depth_sub = Subscriber('/theia/depth', Image)
        
        # Synchronize subscribers
        self.sync = ApproximateTimeSynchronizer(
            [self.image_sub, self.depth_sub],
            queue_size=5,
            slop=0.1
        )
        self.sync.registerCallback(self.synced_callback)
        
        # Publishers
        self.mask_pub = rospy.Publisher('/leaves_masks', masks, queue_size=5)
        self.viz_pub = rospy.Publisher('/leaf_visualization', Image, queue_size=5)

    def init_model(self):
        """Initialize YOLOv8-X model"""
        try:
            # Load pretrained YOLOv8-X model
            model_path = os.path.join(HOME_DIR, 'ultralytics/large_best.pt')
            model = YOLO(model_path)
            
            # Convert to TensorRT if possible
            if torch.cuda.is_available():
                try:
                    from torch2trt import torch2trt
                    dummy_input = torch.randn(1, 3, self.receptive_height, self.receptive_width).cuda()
                    model.model = torch2trt(model.model, [dummy_input])
                    print("TensorRT optimization successful!")
                except Exception as e:
                    print(f"TensorRT optimization failed: {e}")
            
            return model
        except Exception as e:
            rospy.logerr(f"Model initialization failed: {e}")
            return None

    def warmup_gpu(self):
        """Warm up GPU with dummy inference"""
        print('Warming up GPU...')
        dummy_image = np.zeros((self.receptive_height, self.receptive_width, self.im_channels))
        for _ in range(3):  # Multiple warmup runs
            self.model(dummy_image, **self.model_config)
        torch.cuda.empty_cache()

    def process_depth(self, depth_img, masks):
        """Process depth information for each mask"""
        depth_info = []
        for mask in masks:
            mask_array = mask.cpu().numpy()
            masked_depth = depth_img * mask_array
            avg_depth = np.mean(masked_depth[masked_depth > 0])
            depth_info.append(avg_depth)
        return depth_info

    def synced_callback(self, img_msg, depth_msg):
        """Process synchronized image and depth data"""
        try:
            # Convert ROS messages to OpenCV format
            cv_img = self.bridge.imgmsg_to_cv2(img_msg, "rgb8")
            depth_img = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")

            # Run YOLOv8-X inference
            results = self.model(cv_img, **self.model_config)
            
            if results[0].masks is None:
                rospy.logwarn("No leaves detected in current frame")
                return

            # Get masks and process depth
            leaves = results[0].masks.data.cpu().numpy()
            depth_info = self.process_depth(depth_img, results[0].masks.data)
            
            # Create aggregated mask
            mask_aggregated = np.zeros((self.im_height, self.im_width))
            print(f'No. of leaves found: {len(leaves)}')

            for j in range(len(leaves)):
                mask_ = leaves[j, :, :]
                mask = cv.resize(mask_, (self.im_width, self.im_height), 
                               interpolation=cv.INTER_NEAREST)
                mask_aggregated = mask_aggregated + mask * (j + 1)
                mask_aggregated[mask_aggregated > j] = j + 1

            # Save visualization
            viz_path = os.path.join(self.output_dir, f"aggregated_masks{self.count}.png")
            plt.imsave(viz_path, mask_aggregated.astype('uint8'))
            self.count += 1

            # Publish results
            self.publish_results(mask_aggregated, depth_info)

            # Clean up GPU memory
            self.cleanup_gpu()

        except Exception as e:
            rospy.logerr(f"Error in callback: {str(e)}")

    def publish_results(self, mask_aggregated, depth_info):
        """Publish processing results"""
        # Publish mask message
        mask_msg = masks()
        mask_msg.imageData = mask_aggregated.astype('uint16').ravel()
        mask_msg.depthData = depth_info
        self.mask_pub.publish(mask_msg)

        # Publish visualization
        viz_msg = self.bridge.cv2_to_imgmsg(
            cv.applyColorMap(mask_aggregated.astype('uint8'), cv.COLORMAP_JET),
            encoding="bgr8"
        )
        self.viz_pub.publish(viz_msg)

    def cleanup_gpu(self):
        """Clean up GPU memory"""
        gc.collect()
        torch.cuda.empty_cache()

def main():
    rospy.init_node('yolov8x_seg', anonymous=False)
    node = YoloXDepthSeg()
    rospy.spin()

if __name__ == '__main__':
    main()