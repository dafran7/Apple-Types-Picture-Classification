# import tools as tl
# import cv2
# import numpy as np
# from pywt import dwt2
# from sklearn.metrics.cluster import entropy
# import skimage.morphology as mp
#
# img = cv2.imread("gold.jpg")
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# row,col = gray.shape
# # rgb_means, rgb_std = cv2.meanStdDev(img)
#
# canvas = np.zeros((col, row, 1), np.uint8)
# for i in range(row):
#     for j in range(col):
#         if (gray[i][j]<220):
#             canvas.itemset((i, j, 0), 255)
#         else:
#             canvas.itemset((i, j, 0), 0)
#
# kernel = np.ones((3,3),np.uint8)
# gray = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, kernel)
#
# for i in range(row):
#     for j in range(col):
#         b,g,r = img[i][j]
#         if (canvas[i][j]==255):
#             img.itemset((i, j, 0), b)
#             img.itemset((i, j, 1), g)
#             img.itemset((i, j, 2), r)
#         else:
#             img.itemset((i, j, 0), 0)
#             img.itemset((i, j, 1), 0)
#             img.itemset((i, j, 2), 0)
#
#
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# cv2.imshow("HSV Image", hsv)
# cv2.imshow("Biner Image", canvas)
# cv2.imshow("Open Image", img)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

from sklearn.datasets import load_iris
iris = load_iris()
print iris['feature_names']
