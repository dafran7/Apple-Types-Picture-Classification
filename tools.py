import numpy as np
import cv2

def CentreClass(b, g, r):
    if b > g:
        if b > r:
            return '0'
        else:
            return '2'
    elif g > b:
        if g > r:
            return '1'
        else:
            return '2'
    elif r > b:
        if r > g:
            return '2'
        else:
            return '1'
    else:
        return '2'


def Energy(coeffs, k):
    return np.sqrt(np.sum(np.array(coeffs[-k]) ** 2)) / len(coeffs[-k])
