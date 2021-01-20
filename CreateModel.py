import pywt
import numpy as np
import cv2
import glob
import os
import time
import random
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import entropy
from sklearn import svm
import cPickle
import tools

model_filename = 'models/clfSVM-sigmoid-ovr.pkl'

fruit_images = []
labels = []
start_time = time.time()
for fruit_dir_path in glob.glob("Buah/Training/*"):
    images = []
    fruit_label = fruit_dir_path.split("\\")[-1]
    #print(fruit_label)
    for image_path in glob.glob(os.path.join(fruit_dir_path, "*.jpg")):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        image = cv2.resize(image, (70, 70))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        images.append(image)
    images = np.array(images)
    for i in random.sample(range(0,images.shape[0]),300):
        image = images[i]
        fruit_images.append(image)
        labels.append(fruit_label)
fruit_images = np.array(fruit_images)
labels = np.array(labels)

print(fruit_images.shape)

label_to_id_dict = {v:i for i,v in enumerate(np.unique(labels))}
id_to_label_dict = {v: k for k, v in label_to_id_dict.items()}

print(id_to_label_dict)

label_ids = np.array([label_to_id_dict[x] for x in labels])


##### Feature Extraction
list_of_vectors = []
for img in fruit_images:
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
    canvas = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, kernel)

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

    #b, g, r = img[row/2-1,col/2-1]
    #center_code = tools.CentreClass(b,g,r)
    list_of_vectors.append([rgb_means[2], rgb_means[1], rgb_means[0],
                            rgb_std[2], rgb_std[1], rgb_std[0],
                            hsv_means[2], hsv_means[1], hsv_means[0],
                            hsv_std[2], hsv_std[1], hsv_std[0],
                            Energy, Entropy])

list_of_vectors = np.array(list_of_vectors)

# X_train, X_test, y_train, y_test = train_test_split(list_of_vectors, label_ids, test_size=0.30, random_state=14)
X_tr = list_of_vectors
y_tr = label_ids


##### Load Data Test ######
fruit_images = []
labels = []
for fruit_dir_path in glob.glob("Buah/Test/*"):
    fruit_label = fruit_dir_path.split("\\")[-1]
    for image_path in glob.glob(os.path.join(fruit_dir_path, "*.jpg")):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        image = cv2.resize(image, (70, 70))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        fruit_images.append(image)
        labels.append(fruit_label)
fruit_images = np.array(fruit_images)
labels = np.array(labels)

label_to_id_dict = {v:i for i,v in enumerate(np.unique(labels))}
id_to_label_dict = {v: k for k, v in label_to_id_dict.items()}
# Data Test CLASS
y_te = np.array([label_to_id_dict[x] for x in labels])


##### Feature Extraction (Data Test)
X_te = []
for img in fruit_images:
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
    canvas = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, kernel)

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
            if edges[i][j] == 255:
                count += 1

    #b, g, r = img[row/2-1,col/2-1]
    #center_code = tools.CentreClass(b,g,r)
    X_te.append([rgb_means[2], rgb_means[1], rgb_means[0],
                            rgb_std[2], rgb_std[1], rgb_std[0],
                            hsv_means[2], hsv_means[1], hsv_means[0],
                            hsv_std[2], hsv_std[1], hsv_std[0],
                            Energy, Entropy])

X_te = np.array(X_te)

model = svm.SVC(C=1.0,kernel='sigmoid',gamma='auto',random_state=14,
                decision_function_shape='ovr',probability=True)

model = model.fit(X_tr, y_tr)
### Save Model
with open(model_filename, 'wb') as save:
    cPickle.dump(model, save)
### Load Model
# with open(model_filename, 'rb') as file:
#     model = cPickle.load(file)

test_predict = model.predict(X_te)

precision = accuracy_score(y_te,test_predict) * 100
print("Accuracy with SVM: {0:.3f}".format(precision))
print("--- %s seconds ---" % (time.time() - start_time))

