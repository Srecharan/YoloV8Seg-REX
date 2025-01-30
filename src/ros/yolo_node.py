#!/usr/bin/env python3

# YOLO -V8 NODE

import sys
import os
from ultralytics import YOLO
import rospy
import numpy as np
from PIL import Image as PIL_image
from matplotlib import pyplot as plt
from sensor_msgs.msg import Image
from message_filters import ApproximateTimeSynchronizer, Subscriber
import time
import torch
import gc
import cv2 as cv
from yoloV8_seg.msg import masks
from datetime import datetime

HOME_DIR = os.path.expanduser('~')

class YoloSeg:
    def __init__(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = os.path.join(HOME_DIR, 'dummy', timestamp)
        os.makedirs(self.output_dir, exist_ok=True)
        self.masked_images_dir = os.path.join(HOME_DIR, 'leaf_grasp_output/Yolo_outputs')
        os.makedirs(self.masked_images_dir, exist_ok=True)

        rospy.Subscriber('/theia/left/image_rect_color', Image, self.image_callback, queue_size=5)
        self.pub = rospy.Publisher('/leaves_masks', masks, queue_size=5)

        self.model = self.init_()
        self.receptive_width = 1088  # 1440
        self.receptive_height = 832  # 1080  # this is because YOLO wants no's divisible by 32 although the im.rows() = 1080

        self.im_width = 1440
        self.im_height = 1080  # this is because YOLO wants no's divisible by 32 although the im.rows() = 1080
        self.im_channels = 3
        self.count = 0

        # model predict args
        self.conf = 0.7
        self.visualize = False
        self.iou = 0.2
        self.save = True
        self.retina_masks = True
        self.hide_conf = True
        self.hide_labels = True
        self.boxes = False
        self.device = 0

        print('Warming the GPU.....')
        dummy_image = np.zeros((self.receptive_height, self.receptive_width, self.im_channels))
        self.model(dummy_image, imgsz=[self.receptive_height, self.receptive_width])
        print('Yolov8-seg model init complete ... waiting for images ....')

    def init_(self):
        rospy.set_param('/yolo_done', False)
        rospy.set_param('/raft_done', False)
        model_name = rospy.get_param('/yoloV8_seg/model')
        # print(model_name)
        model = YOLO(HOME_DIR +
            '/ultralytics/large_best.pt')  # trained model
        print('init_- success.........')
        # model = YOLO(model_name)  # trained model
        return model

    # parse inpout here
    def image_callback(self, image):

        #proc_start = rospy.get_param("start_proc")
        #while proc_start == False:
           # print(f"\rWating for proc toggle...", end="")
         #   sys.stdout.flush()
          #  proc_start = rospy.get_param("start_proc")

        # print('inside the sync timed callback...')
        n_channels = self.im_channels  # default

        print('image encoding: ', image.encoding)

        if image.encoding == 'rgb8':
            n_channels = self.im_channels  # color image
        else:
            n_channels = 1  # mono image

        # print('im height: ', image.height, ' width: ', image.width)

        img_ = np.ndarray(shape=(image.height, image.width, n_channels), dtype=np.uint8, buffer=image.data)
        img = np.array(img_)
        img = np.flip(img, axis=2)  # why do Ihave to do this in live cam mode but not while playing bag files?
        # print('np im shape: ', img.shape)
        # cv.namedWindow('check', cv.WINDOW_NORMAL)
        # cv.imshow('check', img)
        # cv.waitKey(2)
        # cv.destroyWindow('check')

        # results = self.model(img, visualize=False, imgsz=[self.im_height, self.im_width])  # no. of class of objects
        # results = self.model(img)  # no. of class of objects
        results = self.model.predict(img, conf=self.conf, visualize=self.visualize, iou=self.iou, save=self.save,
                                     retina_masks=self.retina_masks, hide_conf=self.hide_conf,
                                     hide_labels=self.hide_labels, boxes=self.boxes, device=self.device)

        # print('output shape: ', results)
        print('Aggrigating the Masks...')

        leaves = results[0].masks.data.cpu().numpy()  # [0] since we only have one class

        mask_aggregated = np.zeros((self.im_height, self.im_width))
        print('No. of leaves found: ', len(leaves))

        for j in range(len(leaves)):  # no. of masks
            mask_ = leaves[j, :, :]
            mask = cv.resize(mask_, (self.im_width, self.im_height), interpolation=cv.INTER_NEAREST)
            # no_pixels = (mask > 0).sum()
            # if no_pixels > 5:
            # mask_aggregated = mask_aggregated + mask * (j + 1)
            mask_aggregated = mask_aggregated + mask * (j + 1)
            mask_aggregated[mask_aggregated > j] = j+1

            print('for index ', j + 1, ' max val = ', np.amax(mask_aggregated).astype('uint8'))
            # leaf_count = leaf_count + 1
        plt.imsave(os.path.join(self.masked_images_dir, f"aggrigated_masks{self.count}.png"), mask_aggregated.astype('uint8'))
        self.count += 1
        print('Freeing GPU Memory...  yolo Node.....')
        # self.model.model.cpu()
        # del self.model.model
        gc.collect()
        torch.cuda.empty_cache()
        # rospy.signal_shutdown('Just once is enough ...')
        time.sleep(1)
        rospy.set_param('yolo_done', True)

        raft_status = rospy.get_param('/raft_done')
        print('/raft_done: ', raft_status)
        rate = rospy.Rate(5)
        print('YOLO publishing....')
        while raft_status == False:
            maks_msg = masks()
            maks_msg.imageData = mask_aggregated.astype('uint16').ravel()
            self.pub.publish(maks_msg)
            raft_status = rospy.get_param('/raft_done')
            rate.sleep()

        time.sleep(1)
        print('Reinitializing GPU with Yolo-V8 Model...')
        self.model = self.init_()

        # imC = cv.applyColorMap(mask_aggregated.astype('uint8'), cv.COLORMAP_JET)
        # cv.namedWindow("aggrigated", cv.WINDOW_NORMAL)
        # cv.imshow("aggrigated", imC)
        # cv.waitKey(0)


def init():
    b = YoloSeg()
    rospy.init_node('yolov8_seg', anonymous=False)
    print("test")
    rospy.spin()


if __name__ == '__main__':
    init()