from ultralytics import YOLO
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import cv2 as cv

model = YOLO('/home/abhi/ultralytics/large_best.pt')
img = '/home/abhi/Downloads/LEFT_F000025.png'

# results = model(img)

results = model.predict(img, conf=0.3, visualize=False, iou=0.2, save=False, retina_masks=True, hide_conf=True,
                        hide_labels=True, boxes=False, device=0)
# results = model(img, visualize=False, imgsz=[832, 1088])   #  (832, 1088), (1088, 1440)
# View results
Img = cv.imread(img)
# Img = np.asarray(Image.open('/home/buggspray/Desktop/temp/2023-09-22-17-03-07/COLOR/RIGHT_F000000.png'))

leaves = results[0].masks.data.cpu().numpy()  # [0] since we only have one class
print('No. of leaves found: ', len(leaves))

# for i in range(len(leaves)):
#     # print('img class: ', Img.type())
#     mask_ = np.asarray(leaves[i, :, :]).astype('uint8')
#     mask = cv.resize(mask_, (1440, 1080), interpolation=cv.INTER_NEAREST)
#     # mask.reshape((1088, 1440))
#     print('image shape: ', mask.shape)
#     plt.imshow(Img)
#     plt.imshow(mask, alpha=0.5)
#     plt.show()

# mask_aggregated = np.zeros((832, 1088))
mask_aggregated = np.zeros((1080, 1440))
leaf_count = 0
# for i in range(len(results)):  # no. of class

# leaves = results[i]
leaf_ = leaves
# print('new shape: ', leaf_.shape)
for j in range(len(leaves)):  # no. of masks
    # leaf_ = leaves.masks.data.cpu().numpy()
    mask_ = leaf_[j, :, :]
    mask = cv.resize(mask_, (1440, 1080), interpolation=cv.INTER_NEAREST)
    no_pixels = (mask > 0).sum()
    # if no_pixels > 5:
    mask_aggregated = mask_aggregated + mask * (j + 1)
    print('for index ', j+1, ' max val = ', np.amax(mask_aggregated).astype('uint8'))
    # print(mask.shape)
    # mask_aggregated[mask_aggregated > j] = 0
    leaf_count = leaf_count + 1

print('mask_aggregated ', mask_aggregated.shape)
# mask_aggregated = np.delete(mask_aggregated, (0, 100, 200, 300, 550, 750, 800, 1000), axis=0)
# I =  Image.fromarray(mask_aggregated)
# print('Image size: ',I.size)
# plt.imshow(I, cmap='jet')
# plt.show()

# imC = cv.applyColorMap(mask_aggregated.astype('uint8'), cv.COLORMAP_JET)
# cv.namedWindow("aggrigated", cv.WINDOW_NORMAL)
# cv.imshow("aggrigated", imC)
# cv.waitKey(0)
