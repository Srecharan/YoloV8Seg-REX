import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import sklearn.metrics.pairwise as pdist
# import scipy.spatial.distance
# import re

# --------------- centroid of each blob --------------------
def get_centroid(blob):
    # ret, thresh = cv.threshold(blob, 30, 255, 0)
    # plt.imshow(blob)
    # plt.show()

    # ret, thresh = cv.threshold(blob, 30, 255, 0)
    contours, hierarchy = cv.findContours(blob, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    print(len(contours))
    centroids = []
    # centroids = np.zeros((len(contours), 2))
    # print(centroids)
    counter = 0
    for c in contours:
        # calculate moments for each contour
        M = cv.moments(c)
        # calculate x,y coordinate of center
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        centroids.append((cX, cY))
        # centroids[counter] = (cX, cY)
        # counter = counter + 1
        # plt.imshow(np.asarray(blob))
        # plt.plot(cX, cY, 'r*')
        # plt.show()
        # cv.waitKey(0)
    # centroids = centroids
    # np.array(re.split("\s+", centroids.replace('[', '').replace(']', '')), dtype=int)
    # print(centroids.shape)
    return centroids


# def f(x, y):
#     return np.sin(np.sqrt(x ** 2 + y ** 2))
#
# x = np.linspace(-6, 6, 30)
# y = np.linspace(-6, 6, 30)
#
# X, Y = np.meshgrid(x, y)
# Z = f(X, Y)

# print('x.shape: ', x.shape)
# print('y.shape: ', y.shape)
# print('X.shape: ', X.shape)
# print('Y.shape: ', Y.shape)
# print('Z.shape: ', Z.shape)

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.contour3D(X, Y, Z, 50, cmap='binary')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')

im = np.array(cv.imread('/home/buggspray/Downloads/SDF_OUT/temp/sdf_FINAL.png', cv.IMREAD_GRAYSCALE))
# plt.imshow(im)
# plt.plot()
# plt.show()

x = np.linspace(0, im.shape[0], im.shape[0])
y = np.linspace(0, im.shape[1], im.shape[1])
X, Y = np.meshgrid(y, x)

# print('x.shape: ', x.shape)
# print('y.shape: ', y.shape)
# print('X.shape: ', X.shape)
# print('Y.shape: ', Y.shape)

Z = im

# print('x.shape: ', x.shape)
# print('y.shape: ', y.shape)
# print('X.shape: ', X.shape)
# print('Y.shape: ', Y.shape)
# print('Z.shape: ', Z.shape)
# print('global minima: ', np.unravel_index(Z.argmin(), Z.shape))
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 100, cmap='jet')
# ax.plot_wireframe(X, Y, Z, cmap='jet')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
fig.show()
plt.show()

# -------------------------------- read mask --------------------------------------
M_ = cv.imread('/home/buggspray/Downloads/SDF_OUT/temp/aggrigated_masks.png', cv.IMREAD_GRAYSCALE)
mask = np.array(M_)
bg_mask = np.amin(mask)  # background picel as we are reading RGB images
# print('mimn: ', bg_mask)
binary_masks = np.where(mask, mask > bg_mask, 1)
res = binary_masks * im
plt.imshow(res)
# plt.show()

min_global = np.unravel_index(im.argmin(), im.shape)
max_global = np.unravel_index(im.argmax(), im.shape)

print('global max: ', max_global)
print('global min: ', min_global)

# for i in range(1, np.amax(mask).astype('uint8')):

each_masks = np.where(mask, mask != bg_mask, 0)
mask_ = mask * each_masks
mask__ = im * each_masks

optimal_leaf_location = np.unravel_index(mask__.argmax(), mask__.shape)
print('amax on entire masks at once: ', np.amax(mask__))
print('amax on entire masks at once index: ', optimal_leaf_location)

plt.imshow(mask__)
plt.plot(optimal_leaf_location[1], optimal_leaf_location[0], 'r*')
plt.show()

coordinates = []
coordinates.append(min_global)
vals = []
for i in np.unique(mask_):
    # print(i)
    if i == 0:
        continue
    else:
        mask_local = np.where(mask, mask == i, 1)
        local_crop = mask_local * im
        # print(np.unravel_index(local_crop.argmax(), local_crop.shape))
        point_ = np.unravel_index(local_crop.argmax(), local_crop.shape)
        val = np.amax(local_crop)
        # plt.plot(point_[1], point_[0], 'r*')
        # plt.imshow(local_crop)
        coordinates.append(point_)
        vals.append(val)
        plt.show()

print('coors: ', coordinates)
print('amin @ mask: ', vals)
print('amax(min): ', np.amax(np.asarray(vals)), ' at index',
      np.unravel_index(np.asarray(vals).argmax(), np.asarray(vals).shape), '  and coordinates: ',
      coordinates[np.unravel_index(np.asarray(vals).argmax(), np.asarray(vals).shape)[
                      0] + 1])  # +1 because we inserted global min at the top
# print(pdist.euclidean_distances(np.asarray(coordinates)))
# print(vals)

leaf_centroids = get_centroid(binary_masks.astype('uint8'))
# ------------------ pairwise distance from minima  ----------------------

B = np.asarray(leaf_centroids).astype('uint8')
# print(B.shape)
B = np.insert(B, 0, values=(min_global[0], min_global[1]), axis=0)  # Python is an F*ed-up language
# print(B.shape)

# print(coordinates)
pdist_B = np.round(np.array(pdist.euclidean_distances(B))).astype('uint8')
# print(pdist_B.shape)
np.savetxt('/home/buggspray/Downloads/SDF_OUT/temp/pdist_to_minima.txt', pdist_B, fmt='%d')

# ------------------ pairwise distance from maxima  ----------------------
A = np.asarray(leaf_centroids).astype('uint8')
# print(B.shape)
A = np.insert(A, 0, values=(max_global[0], max_global[1]), axis=0)  # Python is an F*ed-up language
# print(B.shape)

# print(coordinates)
pdist_A = np.round(np.array(pdist.euclidean_distances(A))).astype('uint8')
# print(pdist_.shape)
np.savetxt('/home/buggspray/Downloads/SDF_OUT/temp/pdist_to_maxima.txt', pdist_A, fmt='%d')
# print(vals)

# leaf_centroids_ = np.insert(leaf_centroids[0], min_global[0], min_global[1], axis=0)

# plt.arrow(max_global[1], min_global[1], np.linalg.norm(max_global[0] - min_global[0]),
#           np.linalg.norm(max_global[1] - min_global[1]))
# plt.show()

# --------------------------- get gradient to compute quiver ----------------------

# img_blur = cv.GaussianBlur(im, (25, 25), 0)
#
# fig, ax = plt.subplots()
# plt.imshow(img_blur, alpha=0.5)
# dx, dy = np.gradient(img_blur / 255)
# # print('grids ', X.shape, Y.shape)
# # print('vals: ', grad_x[500][500] / 255, ' ,', grad_y[500][500] / 255)
# jump = 13
# ax.quiver(X[::jump, ::jump], Y[::jump, ::jump], dx[::jump, ::jump], dy[::jump, ::jump])
#
# plt.show()

