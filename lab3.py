# -*- coding: utf-8 -*-
"""hw3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1gblsDx02Xo-R1Wzlg8nb_ylLAMnXG9Oa
"""
import cv2
import random
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from math import floor

def SSD(des1, des2):
    return (((des1-des2)**2).sum())**0.5

threshold_fm = 2500
threshold_h = 1000

img1 = cv2.imread('hill1.JPG')
img2 = cv2.imread('hill2.JPG')

match_image = cv2.hconcat([img1, img2])

gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()

kp1, des1 = sift.detectAndCompute(gray1,None)
kp2, des2 = sift.detectAndCompute(gray2,None)

first_match_index = 0
second_match_index = 0
match_list = []
for i in range(len(des1)):
    for j in range(len(des2)):

        if SSD(des1[i], des2[j]) == 0:
            first_match_index = j
        elif SSD(des1[i], des2[j]) > threshold_fm :
            continue
        elif SSD(des1[i], des2[j]) < SSD(des1[i],des2[first_match_index]):
            first_match_index = j
        elif SSD(des1[i], des2[j]) < SSD(des1[i],des2[second_match_index]):
            second_match_index = j

    if SSD(des1[i],des2[first_match_index]) < 0.4*SSD(des1[i], des2[second_match_index]):
        match_list.append([i, first_match_index])

match_index = []
for c, m in enumerate(match_list):
    kp2_point = np.asarray(kp2[m[1]].pt).astype(int)
    kp2_point[0] = kp2_point[0] + img1.shape[1]
    kp2_point = tuple(kp2_point)
    kp1_point =  tuple(np.asarray(kp1[m[0]].pt).astype(int))
    r = random.randint(0,255)
    g = random.randint(0,255)
    b = random.randint(0,255)
    cv2.line(match_image, kp1_point, kp2_point, (r, g, b), 2)
    match_index.append([kp1_point, tuple(np.asarray(kp2[m[1]].pt).astype(int))])

cv2.imwrite('image.jpg', match_image)
plt.imshow(match_image),plt.show()


def get_H(img1_points, img2_points, sample):

    point_n = len(img1_points)
    obj_array = np.zeros((point_n*3, 8))
    b = np.zeros(point_n*3)
    for c, j in enumerate(sample):
        img1p = img1_points[j]
        img2p =  img2_points[j]
        j = j*3
        obj_array[j, 0] = -img1p[0]
        obj_array[j, 1] = -img1p[1]
        obj_array[j, 2] = -1
        obj_array[j, 6] = img1p[0] * img2p[0]
        obj_array[j, 7] = img1p[1] * img2p[0]
        #obj_array[j, 8] = img2p[0]
        b[j] = -img2p[0]

        obj_array[j+1, 3] = -img1p[0]
        obj_array[j+1, 4] = -img1p[1]
        obj_array[j+1, 5] = -1
        obj_array[j+1, 6] = img1p[0] * img2p[1]
        obj_array[j+1, 7] = img1p[1] * img2p[1]
        #obj_array[j+1, 8] = img2p[1]
        b[j+1] = -img2p[1]

        obj_array[j+2, 6] = -img1p[0]
        obj_array[j+2, 7] = -img1p[1]
        #obj_array[j+2, 8] = 1
        b[j+2] = 1

    x = np.ones(9)
    x[:8] = np.linalg.pinv(obj_array).dot(b)
    h = x.reshape(3, 3)

    return h

def homomat(points_in_img1, points_in_img2):
    max_count = 0

    for i in range(100):
        sample = random.sample(range(0, len(points_in_img1)), 10)
        H = get_H(points_in_img1, points_in_img2, sample)

        count = 0
        for i in range(len(points_in_img1)):
            point1 = np.ones(3)
            point1[0] = points_in_img1[i][0]
            point1[1] = points_in_img1[i][1]
            tmp = H.dot(point1)
            point2 = tmp[:1] / tmp[2]
            error = SSD(points_in_img2[i], point2)
            if error < threshold_h:
                count += 1

        if count > max_count:
            optimal_H = H
            max_count = count

    return optimal_H

points_in_img1 = []
points_in_img2 = []

for c, m in enumerate(match_index):
    #points_in_img1.append(kp1[good[i][0].queryIdx].pt)
    #points_in_img2.append(kp2[good[i][0].trainIdx].pt)
    points_in_img1.append(m[0])
    points_in_img2.append(m[1])

H = homomat(points_in_img1, points_in_img2)
H2 = homomat(points_in_img2, points_in_img1)
from skimage.measure import ransac

# group corresponding points
data = np.vstack((gray1,gray2))

# compute H and return
#src_pts = np.float32([ kp1[m.queryIdx].pt for m in match_points ]).reshape(-1,1,2)
#dst_pts = np.float32([ kp2[m.trainIdx].pt for m in match_points ]).reshape(-1,1,2)
M, mask = cv2.findHomography(np.asarray(points_in_img1), np.asarray(points_in_img2), cv2.RANSAC,5.0)
M2, mask = cv2.findHomography(np.asarray(points_in_img2), np.asarray(points_in_img1), cv2.RANSAC,5.0)
print(H)
print(M)

def warp(img1, img2, H):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    pts = []
    pts_value = []
    bound = []
    # compute boundary of new_img2
    bound.append([0, 0, 1])
    bound.append([w2, 0, 1])
    bound.append([0, h2, 1])
    bound.append([w2, h2, 1])
    bound = np.array(bound)
    new_bound =  (H @ bound.T).T

    new_bound[0] = new_bound[0]/new_bound[0][2]
    new_bound[1] = new_bound[1]/new_bound[1][2]
    new_bound[2] = new_bound[2]/new_bound[2][2]
    new_bound[3] = new_bound[3]/new_bound[3][2]

    x_min = int(min(new_bound[0][0], new_bound[1][0], new_bound[2][0], new_bound[3][0]))
    x_max = int(max(new_bound[0][0], new_bound[1][0], new_bound[2][0], new_bound[3][0]))
    y_min = int(min(new_bound[0][1], new_bound[1][1], new_bound[2][1], new_bound[3][1]))
    y_max = int(max(new_bound[0][1], new_bound[1][1], new_bound[2][1], new_bound[3][1]))
    y_max = int(max(y_max, h1))
    x_max = int(max(x_max, w1))
    # compute corresponding points
    for i in range(w2):
        for j in range(h2):
            pts.append([i, j, 1])
            pts_value.append(img2[j, i])
    l = len(pts)
    pts = np.array(pts)
    new_pts = (H @ pts.T).T
    img3 = np.zeros((int(y_max)+1, int(x_max)+1, 3))
    for i in range(l):
        x, y, s = new_pts[i][0], new_pts[i][1], new_pts[i][2]
        x = int(floor((x/s)))
        y =int(floor((y/s)))
        if y >= 0:
            img3[y, x] = pts_value[i]/255

    for i in range(1, int(y_max)):
        for j in range(1, int(x_max)):
             if img3[i, j, 0] == 0:
                img3[i, j] = use_nearest_avg(i, j, img3)

    img3=img3*255
    shape = np.array(img1.shape)
    shape[1] = x_max+1
    subA=np.zeros(shape)
    subB=np.zeros(shape)
    mask=np.zeros(shape)
    subA[:, :img1.shape[1]]=img1
    subB=img3[:img1.shape[0],:]
    mask[:,:img1.shape[1]-(img1.shape[1]-x_min)// 2]=1
    level=8
    GP=[mask]
    for i in range(level - 1):
        GP.append(cv2.pyrDown(GP[i]))
    LP1=[]
    LP2=[]
    imgA=subA
    imgB=subB
    for i in range(level - 1):
        next_imgA = cv2.pyrDown(imgA)
        next_imgB = cv2.pyrDown(imgB)
        imgA_=cv2.pyrUp(next_imgA, dstsize=imgA.shape[1::-1])
        imgB_=cv2.pyrUp(next_imgB, dstsize=imgB.shape[1::-1])
        LP1.append(imgA - imgA_)
        LP2.append(imgB - imgB_)
        imgA = next_imgA
        imgB = next_imgB
    LP1.append(imgA)
    LP2.append(imgB)
    blended = []
    for i, mask in enumerate(GP):
        blended.append(LP1[i] * mask + LP2[i] * (1.0 - mask))
    img4 =blended[-1]
    for lev_img in blended[-2::-1]:
        img4=cv2.pyrUp(img4, dstsize=lev_img.shape[1::-1])
        img4+=lev_img
    img4[img4>255]=255
    img4[img4<0]=0

    plt.figure()
    plt.imshow(img1)
    plt.figure()
    plt.imshow(img2)
    result=img3/255
    result[0:h1, 0:w1] = img1/255
    plt.figure()
    plt.imshow(result)
    result[:img4.shape[0],:]=img4/255
    plt.figure()
    plt.imshow(result)

def use_nearest_avg(i, j, img):
    near = img[i-1, j], img[i+1, j], img[i, j-1], img[i, j+1]
    total = np.zeros(3)
    num = 0
    for pt in near:
        if pt[0] != 0:
            num += 1
            total += pt
    if num != 0:
        return total/num
    else:
        return np.zeros(3)

img1 = cv2.imread('hill1.JPG')
img2 = cv2.imread('hill2.JPG')
warp(img1, img2, M2)