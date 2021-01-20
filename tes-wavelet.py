import pywt
import cv2
import matplotlib.pyplot as plt
import numpy as np


def plotCoeffsImage(data, outdir, level):
    ax1 = plt.subplot(111)
    ax1.imshow(data, cmap='gray')
    plt.title('level' + level)
    plt.savefig(outdir+'_contour_'+level+'.jpg')
    plt.close()
    plt.close('all')
    plt.gcf().clear()


def plotCoeffsSpec(data, outdir, level):
    rows = len(data)
    cols = len(data[0])
    # print (rows,cols)
    # spectrum ###########
    plt.plot(np.ravel(data))
    plt.title('level_' + level + '_' + outdir)
    plt.xlim(0, rows*cols)
    plt.xlabel('Position')
    plt.ylabel('Frequency')

    plt.savefig(outdir+'.jpg')
    plt.close()
    plt.close('all')
    plt.gcf().clear()


def combineFigSpec(ll, lh, hl, hh, level):
    imgLL = cv2.imread(ll)
    imgLH = cv2.imread(lh)
    imgHL = cv2.imread(hl)
    imgHH = cv2.imread(hh)
    imHor1 = np.concatenate([imgLL, imgLH], axis=1)
    imHor2 = np.concatenate([imgHL, imgHH], axis=1)
    imVer = np.concatenate([imHor1, imHor2])
    cv2.imwrite(name + level+'_spectrum.jpg', imVer)


name = '1Wavelets'
image = cv2.imread("bb.jpg")
# image = 255 - image
# hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
# h,s,v = cv2.split(hsv)
# b,g,r = cv2.split(image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
coeff = pywt.dwt2(gray, "haar")  # ---- Dekomposisi lv 1
LL, (LH, HL, HH) = coeff

a = np.concatenate(([LL, LH]), axis=1)
b = np.concatenate(([HL, HH]), axis=1)
c = np.concatenate(([a, b]))
# # print (c)
plotCoeffsImage(c, name, '1')
plotCoeffsSpec(LL, 'LL', '1')
plotCoeffsSpec(LH, 'LH', '1')
plotCoeffsSpec(HL, 'HL', '1')
plotCoeffsSpec(HH, 'HH', '1')
combineFigSpec('LL.jpg', 'LH.jpg', 'HL.jpg', 'HH.jpg', '1')

Energy = (LH**2 + HL**2 + HH**2).sum()/image.size

print Energy
#
#
# coeff = pywt.dwt2(LL,"haar") #---- Dekomposisi lv 2
# LL,(LH,HL,HH) = coeff
# a = np.concatenate(([LL, LH]), axis=1)
# b = np.concatenate(([HL, HH]), axis=1)
# c = np.concatenate(([a, b]))
# # print (c)
# plotCoeffsImage(c, name ,'2')
# plotCoeffsSpec(LL,'ll','2')
# plotCoeffsSpec(LH,'lh','2')
# plotCoeffsSpec(HL,'hl','2')
# plotCoeffsSpec(HH,'hh','2')
# combineFigSpec('ll.jpg','lh.jpg','hl.jpg','hh.jpg','2')
#
# coeff = pywt.dwt2(LL,"haar") #---- Dekomposisi lv 3
# LL,(LH,HL,HH) = coeff
# a = np.concatenate(([LL, LH]), axis=1)
# b = np.concatenate(([HL, HH]), axis=1)
# c = np.concatenate(([a, b]))
# # print (c)
# plotCoeffsImage(c, name ,'3')
# plotCoeffsSpec(LL,'ll','3')
# plotCoeffsSpec(LH,'lh','3')
# plotCoeffsSpec(HL,'hl','3')
# plotCoeffsSpec(HH,'hh','3')
# combineFigSpec('ll.jpg','lh.jpg','hl.jpg','hh.jpg','3')
