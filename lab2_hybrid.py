import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image

def load_img(path):
    img = plt.imread(path)
    #return img[:, :, 0]
    return img


def gaussian_filter(width, height, sigma):
    ret = np.zeros((width, height))
    mu_x = width/2
    mu_y = height/2
    for i in range(width):
        for j in range(height):
            ret[i, j] = math.exp(-1.0 * ((i-mu_x)**2 + (j-mu_y)**2) / (2 * sigma**2))
    return ret, 1-ret

def ideal_filter(width, height, D):
    ret = np.zeros((width, height))
    mu_x = width/2
    mu_y = height/2
    for i in range(width):
        for j in range(height):
            if ((i-mu_x)**2 + (j-mu_y)**2) <= D**2:
                ret[i, j] = 1
            else:
                ret[i, j] = 0
    return ret, 1-ret

def shift(img):
    M = img.shape[0]
    N = img.shape[1]
    ret = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            ret[i][j] = img[i][j]*(-1)**(i+j)
    return ret


def filter_img(img, filt):
    img = shift(img)
    img_ft = np.fft.fft2(img)
    ft_low = (img_ft) * filt
    ret = np.fft.ifft2(ft_low)
    ret = np.real(ret)
    ret = shift(ret)
    return np.real(ret)


def hybrid(img_low, img_high, filt_low, filt_high):
    filter_low = np.zeros((img_low.shape[0], img_low.shape[1], 3))
    filter_high = np.zeros((img_low.shape[0], img_low.shape[1], 3))
    for i in range(3):
        filter_low[:, :, i] = filter_img(img_low[:, :, i], filt_low)
        filter_high[:, :, i] = filter_img(img_high[:, :, i], filt_high)
    return filter_low + filter_high


path_high = './hw2_data/task1and2_hybrid_pyramid/5_fish.bmp'
path_low = './hw2_data/task1and2_hybrid_pyramid/5_submarine.bmp'
img_high = load_img(path_high)
img_low = load_img(path_low)
plt.figure()
plt.axis('off')
plt.imshow(img_high)
plt.figure()
plt.axis('off')
plt.imshow(img_low)
filt_low, filt_high = ideal_filter(img_low.shape[0], img_low.shape[1], 12)
ret = hybrid(img_low, img_high, filt_low, filt_high)
plt.figure()
plt.axis('off')
plt.imshow(ret.astype(int))
filt_low, filt_high = gaussian_filter(img_low.shape[0], img_low.shape[1], 12)
ret = hybrid(img_low, img_high, filt_low, filt_high)
plt.figure()
plt.axis('off')
plt.imshow((ret.astype(int)))