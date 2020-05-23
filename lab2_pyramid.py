#from google.colab import files
#uploaded = files.upload()

import numpy as np
import glob
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import math

def gaussian_kernel_convolution(img):
    sigma=1
    kernel_size=5
    x,y=np.mgrid[-(kernel_size//2):kernel_size//2+1,-(kernel_size//2):kernel_size//2+1]
    gaussian_kernel=np.exp(-pow(x,2)-pow(y,2))
    gaussian_kernel=gaussian_kernel/gaussian_kernel.sum()
    #print(np.shape(img))
    new_img=np.zeros((np.shape(img)[0]//2,np.shape(img)[1]//2))
    #print(np.shape(new_img))
    for i in range (np.shape(new_img)[0]):
        for j in range(np.shape(new_img)[1]):
            for p in range(kernel_size):
                for q in range(kernel_size):
                    x=2*i+1-p+kernel_size//2
                    y=2*j+1-q+kernel_size//2
                    if x<0:
                        #x=x+np.shape(img)[0]
                        x=0
                    elif x>=np.shape(img)[0]:
                        #x=x-np.shape(img)[0]
                        x=np.shape(img)[0]-1
                    if y<0:
                        #y=y+np.shape(img)[1]
                        y=0
                    elif y>=np.shape(img)[1]:
                        #y=y-np.shape(img)[1]
                        y=np.shape(img)[1]-1
                    new_img[i][j]+=gaussian_kernel[p][q]*img[x][y]
            new_img[i][j]=int(new_img[i][j])
    return new_img

def upsample(img):
    new_img=np.zeros((np.shape(img)[0]*2,np.shape(img)[1]*2))
    tmp_img=np.zeros((np.shape(img)[0]*2,np.shape(img)[1]*2))#for duplicate image
    sigma=1
    kernel_size=5
    x,y=np.mgrid[-(kernel_size//2):kernel_size//2+1,-(kernel_size//2):kernel_size//2+1]
    gaussian_kernel=np.exp(-pow(x,2)-pow(y,2))
    gaussian_kernel=gaussian_kernel/gaussian_kernel.sum()
    #gaussian_kernel=gaussian_kernel*4
    for i in range(np.shape(img)[0]):
       for j in range(np.shape(img)[1]):
          tmp_img[2*i+1][2*j+1]=img[i][j]
          tmp_img[2*i][2*j+1]=img[i][j]
          tmp_img[2*i][2*j]=img[i][j]
          tmp_img[2*i+1][2*j]=img[i][j]
    for i in range(np.shape(new_img)[0]):
        for j in range(np.shape(new_img)[1]):
            for p in range(kernel_size):
                for q in range(kernel_size):
                    x=i-p+kernel_size//2
                    y=j-p+kernel_size//2
                    if x<0:
                        x=0
                    elif x>=np.shape(new_img)[0]:
                        x=np.shape(new_img)[0]-1
                    if y<0:
                        y=0
                    elif y>=np.shape(new_img)[1]:
                        y=np.shape(new_img)[1]-1
                    new_img[i][j]+=gaussian_kernel[p][q]*tmp_img[x][y]
    return new_img

#def laplacian_pyramid(img):
    

def cv2_pyramid(img,layer=5):
    pyr=[]
    pyr.append(img)
    for i in range(layer):
        img=cv2.pyrDown(img)
        pyr.append(img)
    return pyr

if __name__ == "__main__":
    images=glob.glob('*.jpg')
    img=cv2.imread(images[1],cv2.IMREAD_GRAYSCALE)
    img=cv2.resize(img,(508,672))
    plt.figure(figsize=(45,30))
    layer=5
    result_down=[]
    result_up=[]
    ans_up=[]
    laplacian_result=[]
    result_down.append(img)
    ans=cv2_pyramid(img)
    for l in range(layer):
        img=gaussian_kernel_convolution(img)
        #cv2.imwrite(str(l+1)+'_'+images[0],img)
        result_down.append(img)
    bottom=result_down[-1]
    result_up.append(bottom)
    ans_up.append(ans[-1])
    laplacian_result.append(bottom)
    laplacian_ans=[]
    laplacian_ans.append(ans[-1])
    for l in range(layer):
        size=np.shape(result_down[layer-l-1])
        bottom=upsample(result_down[layer-l])
        result_up.append(bottom)
        #print(np.shape(result_down[layer-1-l]),np.shape(bottom))
        laplacian = np.zeros(size)
        for i in range(size[0]):
            for j in range(size[1]):
                if i>=np.shape(bottom)[0] or j>=np.shape(bottom)[1]:
                    laplacian[i][j]=0
                else:
                    laplacian[i][j]=result_down[layer-l-1][i][j]-bottom[i][j]
        laplacian_result.append(laplacian)
                
        #laplacian.append(result_down[layer-l-1]-bottom)
        size = (ans[layer-l-1].shape[1],ans[layer-l-1].shape[0])
        bottom = cv2.pyrUp(ans[layer-l], dstsize=size)
        ans_up.append(bottom)
        laplacian = cv2.subtract(ans[layer-l- 1], bottom)
        laplacian_ans.append(laplacian)
    

    for i in range(layer+1):
        plt.subplot(6,layer+1,1+i)
        plt.imshow(result_down[i],'gray')
        #print(np.shape(ans[i]))
        plt.subplot(6,layer+1,layer+2+i)
        plt.imshow(result_up[layer-i],'gray')
        #print(np.shape(result_down[i]))
        #print(np.shape(result_up[layer-i]))
        plt.subplot(6,layer+1,2*layer+3+i)
        plt.imshow(laplacian_result[layer-i],'gray')  
        plt.subplot(6,layer+1,3*layer+4+i)
        plt.imshow(ans[i],'gray')
        #print(np.shape(ans_up[layer-i]))
        plt.subplot(6,layer+1,4*layer+5+i)
        plt.imshow(ans_up[layer-i],'gray')
        plt.subplot(6,layer+1,5*layer+6+i)
        plt.imshow(laplacian_ans[layer-i],'gray')
        
        
images=glob.glob('*.jpg')
print(images)

f_conv=[ ]
f_lap=[ ]
layer=5
plt.figure(figsize=(30,20))
for l in range(layer+1):
    
    #FOURIER TRANSFORM FOR GAUSSIAN 
    f = np.fft.fft2(result_down[l])
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    f_conv.append(magnitude_spectrum)

    #FOURIER TRANSFORM FOR LAPLACIAN
    f = np.fft.fft2(laplacian_result[layer-l])
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    f_lap.append(magnitude_spectrum)

for i in range(layer+1):
    plt.subplot(2,layer+1,1+i)
    plt.imshow(f_conv[i])
    plt.subplot(2,layer+1,layer+2+i)
    plt.imshow(f_lap[i])