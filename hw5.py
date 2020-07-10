from libsvm.svmutil import *
import cv2
import random
import numpy as np
from matplotlib import pyplot as plt
import sys
from PIL import Image
import os
import glob
from collections import Counter
from cyvlfeat.kmeans import kmeans
from cyvlfeat.sift.sift import sift
from cyvlfeat.sift.dsift import dsift
import scipy.io as sio
import pickle
import scipy.spatial.distance as distance



label=['bedroom','coast','forest','highway','industrial','insidecity','kitchen','livingroom'
            ,'mountain','office','opencountry','store','street','suburb','tallbuilding']

def read_image(mode='train'):
    x=[]
    y=[]
    if mode=='train':
        print("---load training data---")
        d=[x[0] for x in os.walk('train')]
        d.sort()
        for label_number,label_name in enumerate(d[1:]):
            print(label_name)
            images=glob.glob(label_name+'/*.jpg')
            for i in range(len(images)):
                img=cv2.imread(images[i],cv2.IMREAD_GRAYSCALE).astype('float64')
                np.reshape(img,(1,-1))
                x.append(img)
                y.append(label_number)
    else:
        d=[x[0] for x in os.walk('test')]
        d.sort()
        print("---load testing data---")
        for label_number,label_name in enumerate(d[1:]):
            print(label_name)
            images=glob.glob(label_name+'/*.jpg')
            for i in range(len(images)):
                img=cv2.imread(images[i],cv2.IMREAD_GRAYSCALE).astype('float64')
                np.reshape(img,(1,-1))
                x.append(img)
                y.append(label_number)
    return x,y

def tiny_image_representation(data,crop=False):
    result=[]
    for i in range(len(data)):
        img=cv2.resize(data[i],(16,16)).astype('float64')
        img-=np.mean(img)
        img=img/np.std(img)
        result.append(img)
    return result#do central crop


def compute_center(data, k):
    des_list=[]
    des_all=[]
    for i in range(len(data)):
        kp,des = dsift(data[i], step=[5,5], fast=True)
        if des is not None:
            des_list.append(des.astype('float64'))
            des_all.extend(des.astype('float64'))
        else:
            print("no feature points")
            des_list.append(np.zeros((1,128)))
    center = kmeans(np.array(des_all).astype("float64"), k, initialization="PLUSPLUS")      
    return center

def bag_of_sift(data, k, center):
    image_feats = np.zeros((len(data), k))
    for i in range(len(data)):
        kp,des = dsift(data[i], step=[9,9], fast=True)
        dist = distance.cdist(center, des, 'euclidean')
        mdist = np.argmin(dist, axis = 0)
        histo, bins = np.histogram(mdist, range(k+1))
        if np.linalg.norm(histo) == 0:
            image_feats[i, :] = histo
        else:
            image_feats[i, :] = histo / np.linalg.norm(histo)   
    return image_feats
    

def knn(input_data,train_data,train_label,k=25):
    #implement two distance function
    distance=[]
    for i in range(len(train_data)):
        dist=np.linalg.norm(train_data[i]-input_data)
        distance.append(dist)
    sorted_distance=sorted(range(len(distance)),key=lambda x:distance[x])
    neighbor_label=[train_label[j] for j in sorted_distance[:k]]
    c=Counter(neighbor_label)
    #print(c)
    return c.most_common(1)[0][0]

##task 1
train_data,train_label=read_image('train')
test_data,test_label=read_image('test')
train_data=tiny_image_representation(train_data)
test_data=tiny_image_representation(test_data)

correct = 0.0
total = 0.0

for idx, testImage in enumerate(test_data):
    predict = knn(testImage, train_data, train_label, 1)
    total = total +1

    if test_label[idx] == predict: 
        correct = correct +1

accuracy = correct / total

print('task 1 accuracy: {}'.format(accuracy))


k = 100
train_data,train_label=read_image('train')
test_data,test_label=read_image('test')
center = compute_center(train_data, k)
with open('center_100.pkl', 'wb') as handle:
    pickle.dump(center, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
train_f = bag_of_sift(train_data, k, center)
with open('train_f_100.pkl', 'wb') as handle:
    pickle.dump(train_f, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
test_f = bag_of_sift(test_data, k, center)
with open('test_f_100.pkl', 'wb') as handle:
    pickle.dump(test_f, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
print("finish doing sift")


# knn
test_data = test_f
train_data = train_f
count = 0
for i in range(len(test_data)):
    predict_label=knn(test_data[i],train_data,train_label, 3)
    #print(test_label[i],predict_label)
    if test_label[i]==predict_label:
        count+=1
print('task 2 accuracy: {}'.format(count/len(test_data)))


for j in range(1, 60):
    count = 0
    for i in range(len(test_data)):
        predict_label=knn(test_data[i],train_data,train_label, j)
        if test_label[i]==predict_label:
            count+=1
    print('k = {}'.format(j))
    print('task 2 accuracy: {}'.format(count/len(test_data)))


def svm(kernel, X_train, Y_train, X_test, Y_test):
    model = svm_train(Y_train, X_train, kernel)
    svm_predict(Y_test, X_test, model)
svm('-t 0 -c 10', train_f, train_label, test_f, test_label)

