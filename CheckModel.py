import pywt
import numpy as np
import cv2
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import glob
import os
import random
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import entropy
from sklearn import svm
import cPickle
import tools

### Load Image
img = cv2.imread("Google/gd1.jpg")
img = cv2.resize(img, (70, 70))
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
X_te = []

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
row,col = gray_img.shape
canvas = np.zeros((col, row, 1), np.uint8)
luas = 0
for i in range(row):
    for j in range(col):
        if gray_img[i][j] < 220:
            canvas.itemset((i, j, 0), 255)
            luas += 1
        else:
            canvas.itemset((i, j, 0), 0)

kernel = np.ones((3,3),np.uint8)
gray = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, kernel)

for i in range(row):
    for j in range(col):
        b,g,r = img[i][j]
        if canvas[i][j] == 255:
            img.itemset((i, j, 0), b)
            img.itemset((i, j, 1), g)
            img.itemset((i, j, 2), r)
        else:
            img.itemset((i, j, 0), 0)
            img.itemset((i, j, 1), 0)
            img.itemset((i, j, 2), 0)

hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
rgb_means, rgb_std = cv2.meanStdDev(img)
hsv_means, hsv_std = cv2.meanStdDev(hsv_img)

coeff = pywt.dwt2(gray_img, "haar")  # ---- Dekomposisi lv 1
LL, (LH, HL, HH) = coeff
Energy = (LH**2 + HL**2 + HH**2).sum()/img.size
Entropy = entropy(gray_img)

edges = cv2.Canny(img,300,300)
count = 0
for i in range(row):
    for j in range(col):
        if (edges[i][j] == 255):
            count += 1

b, g, r = img[row/2-1,col/2-1]
center_code = tools.CentreClass(b,g,r)
X_te.append([rgb_means[2], rgb_means[1], rgb_means[0],
            rgb_std[2], rgb_std[1], rgb_std[0],
            hsv_means[2], hsv_means[1], hsv_means[0],
            hsv_std[2], hsv_std[1], hsv_std[0],
            Energy, Entropy, count, center_code])

### Load Model
with open('models/clfMLP-sgd1010-0.15-50.pkl', 'rb') as file:
    model = cPickle.load(file)

test_predict = model.predict(X_te)
print("FRUIT CLASS : "+str(test_predict[0]))
