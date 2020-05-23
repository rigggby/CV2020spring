#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import math
import numpy as np
import skimage as sk
import skimage.io as skio
from tqdm import tqdm


# In[2]:


def sobel(img):
    width = img.shape[0]
    height = img.shape[1]
    c1 = np.array([1,0,-1], dtype=np.float)
    c2 = np.array([1,2,1], dtype=np.float)
    out_v = np.zeros((width, height))
    out_h = np.zeros((width, height))
    
    for i in range(height):
        out_v[:,i] = np.convolve(c1, img[:,i], "same")
    for j in range(width):
        out_v[j,:] = np.convolve(c2, out_v[j,:], "same")
        
    for i in range(height):
        out_h[:,i] = np.convolve(c2, img[:,i], "same")
    for j in range(width):
        out_h[j,:] = np.convolve(c1, out_h[j,:], "same")
    
    
    out = (out_v**2 + out_h**2)**0.5
    return out


# In[3]:


def gaussian_kernel_convolution(img):
    sigma=1
    kernel_size=5
    x,y=np.mgrid[-(kernel_size//2):kernel_size//2+1,-(kernel_size//2):kernel_size//2+1]
    gaussian_kernel=np.exp(-pow(x,2)-pow(y,2))
    gaussian_kernel=gaussian_kernel/gaussian_kernel.sum()
    new_img=np.zeros((np.shape(img)[0]//2,np.shape(img)[1]//2))
    
    for i in range (np.shape(new_img)[0]):
        for j in range(np.shape(new_img)[1]):
            for p in range(kernel_size):
                for q in range(kernel_size):
                    x=2*i+1-p+kernel_size//2
                    y=2*j+1-q+kernel_size//2
                    if x<0:
                        x=0
                    elif x>=np.shape(img)[0]:
                        x=np.shape(img)[0]-1
                    if y<0:
                        y=0
                    elif y>=np.shape(img)[1]:
                        y=np.shape(img)[1]-1
                    new_img[i][j]+=gaussian_kernel[p][q]*img[x][y]
            new_img[i][j]=int(new_img[i][j])
            
    return new_img


# In[4]:


def pyramidDown(img):
    s = np.std(img)
    img = gaussian_filter(img, sigma=s)
    img_out = img[::2, ::2]
    
    return img_out


# In[5]:


def alignImgs_gauss_pyr(img1, img2):

    winSize = 5

    img2_align = img2
    img1_origin = img1
    
    ite = int(math.log(img1.shape[0] / 32 , 2)) 
    x_shift, y_shift = (0, 0)
    x, y = (0,0)
    
    gauss_pyr1 = []
    gauss_pyr2 = []
    for i in tqdm(range(ite)):
        #img1 = pyramidDown(img1)
        #img2 = pyramidDown(img2)
        img1 = gaussian_kernel_convolution(img1)
        img2 = gaussian_kernel_convolution(img2)
        gauss_pyr1.append(img1)
        gauss_pyr2.append(img2)


    for k in range(ite):
        
        img1 = gauss_pyr1[k]
        img2 = gauss_pyr2[k]
        img2 = np.roll(img2, int(x_shift/(2**(ite-k))), axis=0)
        img2 = np.roll(img2, int(y_shift/(2**(ite-k))), axis=1)
        x, y = (0,0)
    
        ssd_min = np.sum((img1-img2)**2)
        for i in tqdm(range(-winSize, winSize)):
            for j in range (-winSize, winSize):
            
                img2_revise = np.roll(img2, i, axis=0)
                img2_revise = np.roll(img2_revise, j, axis=1)

                ssd = np.sum((img1 - img2_revise)**2)
                if ssd < ssd_min:
                    x, y = (i, j)
                    ssd_min = ssd

        img2_align = np.roll(img2_align, x*(2**(ite-k-1)), axis=0)
        img2_align = np.roll(img2_align, y*(2**(ite-k-1)), axis=1)
        x_shift = x_shift + x*(2**(ite-k-1))
        y_shift = y_shift + y*(2**(ite-k-1))
    
    return x_shift, y_shift


# In[6]:


def gaussian_filter(img, sigma):
    width = img.shape[0]
    height = img.shape[1]
    out = np.zeros((width, height))
    
    #radius = 4*sigma +0.5
    radius = 5
    x = np.arange(-radius, radius+1)
    phi_x = np.exp(-0.5 / (sigma * sigma) * x ** 2)
    w = phi_x / phi_x.sum()
    
    for i in range(height):
        out[:,i] = np.convolve(w, img[:,i], "same")
    for j in range(width):
        out[j,:] = np.convolve(w, out[j,:], "same")
    
    return out


# In[7]:


def alignImgs_brute_force(img1, img2):

    winSize = 50
    x, y = (0,0)
   
    ssd_min = np.sum((img1-img2)**2)
    for i in tqdm(range(-winSize, winSize)):
        for j in range (-winSize, winSize):

            img2_revise = np.roll(img2, i, axis=0)
            img2_revise = np.roll(img2_revise, j, axis=1)

            ssd = np.sum((img1 - img2_revise)**2)
            
            if ssd < ssd_min:
                x, y = (i, j)
                ssd_min = ssd
    
    return x, y


# In[8]:


def thresholding(img):
    img_mean = img.mean()
    img_original = img 
    img[img < img.mean()] = 0
    img = ((img * 4 + img_original)*255).astype(np.uint8)
    
    return img


# In[9]:


def alignImgs_gauss_pyr_with_s(img1, img2):

    winSize = 5

    img2_align = img2
    img1_origin = img1
    ite = int(math.log(img1.shape[0] / 32 , 2)) 
    x_shift, y_shift = (0, 0)
    x, y = (0,0)
    
    gauss_pyr1 = []
    gauss_pyr2 = []
    for i in tqdm(range(ite)):
        img1 = pyramidDown(img1)
        img2 = pyramidDown(img2)
        #img1 = gaussian_kernel_convolution(img1)
        #img2 = gaussian_kernel_convolution(img2)
        gauss_pyr1.append(img1)
        gauss_pyr2.append(img2)


    for k in range(ite):
        
        img1 = gauss_pyr1[k]
        img2 = gauss_pyr2[k]
        img2 = np.roll(img2, int(x_shift/(2**(ite-k))), axis=0)
        img2 = np.roll(img2, int(y_shift/(2**(ite-k))), axis=1)
        
        
        x, y = (0,0)
     
        s1 = np.std(img1)
        s2 = np.std(img2)
        img1_l = sobel(img1) 
        img2_l = sobel(img2) 
        img1_l = img1 + 2*img1_l
        img2_l = img2 + 2*img2_l


        ssd_min = np.sum((img1_l-img2_l)**2) 
        for i in tqdm(range(-winSize, winSize)):
            for j in range (-winSize, winSize):
            
                img2_revise = np.roll(img2_l, i, axis=0)
                img2_revise = np.roll(img2_revise, j, axis=1)

                ssd = np.sum((img1_l - img2_revise)**2)
                if ssd < ssd_min:
                    x, y = (i, j)
                    ssd_min = ssd

        x_shift = x_shift + x*(2**(ite-k-1))
        y_shift = y_shift + y*(2**(ite-k-1))
    
    
    return x_shift, y_shift


# In[22]:


def alignImgs_lap_pyr(img1, img2):

    winSize = 5

    img2_align = img2
    img1_origin = img1
    ite = int(math.log(img1.shape[0] / 32 , 2)) 
    x_shift, y_shift = (0, 0)
    x, y = (0,0)
    
    gauss_pyr1 = []
    gauss_pyr2 = []
    for i in tqdm(range(ite)):
        img1 = pyramidDown(img1)
        img2 = pyramidDown(img2)
        #img1 = gaussian_kernel_convolution(img1)
        #img2 = gaussian_kernel_convolution(img2)
        gauss_pyr1.append(img1)
        gauss_pyr2.append(img2)


    for k in range(ite):
        
        img1 = gauss_pyr1[k]
        img2 = gauss_pyr2[k]
        img2 = np.roll(img2, int(x_shift/(2**(ite-k))), axis=0)
        img2 = np.roll(img2, int(y_shift/(2**(ite-k))), axis=1)
        
        
        x, y = (0,0)
     
        s1 = np.std(img1)
        s2 = np.std(img2)
        
        img1_l = thresholding(gaussian_filter(img1, sigma=5))
        img2_l = thresholding(gaussian_filter(img2, sigma=5))
        img1_l = thresholding(img1 - img1_l)
        img2_l = thresholding(img2 - img2_l)

        ssd_min = np.sum((img1_l-img2_l)**2) 
        for i in tqdm(range(-winSize, winSize)):
            for j in range (-winSize, winSize):
            
                img2_revise = np.roll(img2_l, i, axis=0)
                img2_revise = np.roll(img2_revise, j, axis=1)

                ssd = np.sum((img1_l - img2_revise)**2)
                if ssd < ssd_min:
                    x, y = (i, j)
                    ssd_min = ssd

        x_shift = x_shift + x*(2**(ite-k-1))
        y_shift = y_shift + y*(2**(ite-k-1))
    
    
    return x_shift, y_shift


# In[25]:


def main(imname):  

    im = skio.imread(imname)

    # convert to double (might want to do this later on to save memory)    
    im = sk.img_as_float(im)
    height = int(np.floor(im.shape[0] / 3.0)) 
    width = im.shape[1]

    # separate color channels
    b = im[:height]
    g = im[height: 2*height]
    r = im[2*height: 3*height]
    b = (b*255).astype(np.uint8)
    g = (g*255).astype(np.uint8)
    r = (r*255).astype(np.uint8)

    g_align = np.zeros((height, width))
    r_align = np.zeros((height, width))
    
    # align the images
    x, y = alignImgs_lap_pyr(b, g)
    g_align = np.roll(g, x, axis=0)
    g_align = np.roll(g_align, y, axis=1)

    x, y = alignImgs_lap_pyr(b, r)
    r_align = np.roll(r, x, axis=0)
    r_align = np.roll(r_align, y, axis=1)
    

    # create a color image
    im_out = np.dstack([r_align, g_align, b])
    fname = './out/' + imname.split('/')[-1] +'_color.jpg'
    skio.imsave(fname, im_out)

    # display the image
    skio.imshow(im_out)
    skio.show()


# In[26]:


import glob, os
os.chdir("/Users/heidicheng/Desktop/heidi/Computer Vision/CV2020_HW2/hw2_data/task3_colorizing/")
for file in glob.glob("*.jpg"):
    main(file)
for file in glob.glob("*.tif"):
    main(file)


# In[ ]:




