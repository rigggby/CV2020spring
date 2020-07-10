import cv2
import random
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def get_co_points(img1, img2):
    
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)

    # As per Lowe's ratio test to filter good matches
    good_matches = []
    match_index_img1 = []
    match_index_img2 = []
    good_kp1 = []
    good_kp2 = []
    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            good_matches.append(m)
            match_index_img1.append(kp1[m.queryIdx].pt)
            match_index_img2.append(kp2[m.trainIdx].pt)
            good_kp1.append(kp1[m.queryIdx])
            good_kp2.append(kp2[m.trainIdx])

    return good_matches, match_index_img1, match_index_img2, good_kp1, good_kp2

def image_co_normalize(match_index_img):
    
    img_normalized = []
    match_index_img = np.asarray(match_index_img)
    
    C = np.ones(3)
    C = np.diag(C)
    C[0,2] = -(np.mean(match_index_img[:, 0]))
    C[1,2] = -(np.mean(match_index_img[:, 1]))
    
    s1 = np.sqrt(2) / np.std(match_index_img[:, 0])
    s2 = np.sqrt(2) / np.std(match_index_img[:, 1])
    S = np.zeros((3,3))
    S[2,2] = 1
    S[0,0] = s1
    S[1,1] = s2
    
    T = np.matmul(S, C)
    
    for i in range(len(match_index_img)):
        x = np.ones(3)
        x[:2] = match_index_img[i]
        x = np.matmul(T, x.T)
        
        img_normalized.append(x)
        
    return img_normalized, T

def get_F(match_index_img1, match_index_img2, T1, T2):
    
    point_n = len(match_index_img1)
    A = np.zeros((point_n, 9))
    
    for i in range(point_n):
        
        A[i][0] = match_index_img2[i][0] * match_index_img1[i][0]
        A[i][1] = match_index_img2[i][0] * match_index_img1[i][1]
        A[i][2] = match_index_img2[i][0]
        A[i][3] = match_index_img2[i][1] * match_index_img1[i][0]
        A[i][4] = match_index_img2[i][1] * match_index_img1[i][1]
        A[i][5] = match_index_img2[i][1]
        A[i][6] = match_index_img1[i][0]
        A[i][7] = match_index_img1[i][1]
        A[i][8] = 1
        
    u, s, v = np.linalg.svd(A)
    f = v.T[:, -1]
    F = f.reshape((3,3))
    
    u, s, v = np.linalg.svd(F)
    s = np.diag(s)
    s[2, 2] = 0
    F = np.matmul(u, np.matmul(s, v))
    F = np.matmul(np.matmul(T1.T, F), T2)
    F = F/F[2,2]
    
    return F 

def get_F_sample(match_index_img1, match_index_img2, sample, T1, T2):
    
    A = np.zeros((len(sample), 9))
    
    for c, i in enumerate(sample):
        A[c][0] = match_index_img2[i][0] * match_index_img1[i][0]
        A[c][1] = match_index_img2[i][0] * match_index_img1[i][1]
        A[c][2] = match_index_img2[i][0]
        A[c][3] = match_index_img2[i][1] * match_index_img1[i][0]
        A[c][4] = match_index_img2[i][1] * match_index_img1[i][1]
        A[c][5] = match_index_img2[i][1]
        A[c][6] = match_index_img1[i][0]
        A[c][7] = match_index_img1[i][1]
        A[c][8] = 1
        
    u, s, v = np.linalg.svd(A)
    f = v.T[:, -1]
    F = f.reshape((3,3))
    
    u, s, v = np.linalg.svd(F)
    s = np.diag(s)
    s[2, 2] = 0
    F = np.matmul(u, np.matmul(s, v))
    F = np.matmul(np.matmul(T1.T, F), T2)
    F = F/F[2,2]
    
    return F 

def point_dis_to_line(point, l):
    return abs(np.matmul(point.T, l)) / np.sqrt(l[0]**2 + l[1]**2)

def get_F_RANSAC(img1_index_norm, img2_index_norm, img1_index, img2_index, T1, T2):
    
    max_count = 0
    for i in range(10000):
        count = 0
        sample = random.sample(range(0, len(img1_index)), 8)
        F = get_F_sample(img1_index_norm, img2_index_norm, sample, T1, T2)
        for j in range(len(img1_index)):
            x1 = np.ones(3)
            x1[:2] = img1_index[j]
            l = np.dot(F.T, x1)
            x2 = np.ones(3)
            x2[:2] = img2_index[j]
            error = point_dis_to_line(x2, l)
            if error < 15:
                count += 1
        
        if(count > max_count):
            optimal_F = F
            max_count = count
        
    return optimal_F

def draw_epipolar(match_index_img1, match_index_img2, F, img):
    w = img.shape[1]
    h = img.shape[0]
    
    for i in range((len(match_index_img1))):
        x = np.ones(3)
        x[:2] = match_index_img1[i]
        l = np.dot(F.T, x)
        
        y_0 = map(int, [0, -l[2]/l[1]]) # rough calculation for end points on epipolar line 
        y_w = map(int, [w, -(w*l[0] + l[2])/l[1]])
        
        r = random.randint(0,255)
        g = random.randint(0,255)
        b = random.randint(0,255)
        cv2.line(img, tuple(y_0), tuple(y_w), (r, g, b), 2)
    #plt.imshow(img),plt.show()
    return img
# Find second camera matrix
# Will return 4 possible P2
def P2(K1, K2, F):
    # E
    E = np.matmul(K1.T, np.matmul(F, K2))
    
    # P1
    P1 = np.zeros((3, 4))
    for i in range(3):
        P1[i][i] = 1
    
    # P2
    P2_list = []
    
    # W
    W = np.array([[0, -1, 0], 
                         [1, 0, 0],
                         [0, 0, 1]])
    
   # there are 4 possible P2
    U, S, Vt = np.linalg.svd(E)
    u3 = U[:, 2]
    m = (S[0] + S[1])/2
    S = np.array([[m, 0, 0], 
                         [0, m, 0],
                         [0, 0, 0]])
    E = np.matmul(U, np.matmul(S, Vt))
    U, S, Vt = np.linalg.svd(E)
    
    # possible 1
    Vt = Vt.T
    P2 = np.zeros((3, 4))
    P2[:, :3] = np.matmul(U, np.matmul(W, Vt))
    P2[:, 3] = u3
    P2_list.append(P2)
    
    # possible 2
    P2 = np.zeros((3, 4))
    P2[:, :3] = np.matmul(U, np.matmul(W, Vt))
    P2[:, 3] = -1*u3
    P2_list.append(P2)
    
    # possible 3
    P2 = np.zeros((3, 4))
    P2[:, :3] = np.matmul(U, np.matmul(W.T, Vt))
    P2[:, 3] = u3
    P2_list.append(P2)
    
    # possible 4
    P2 = np.zeros((3, 4))
    P2[:, :3] = np.matmul(U, np.matmul(W.T, Vt))
    P2[:, 3] = -1*u3
    P2_list.append(P2)
    
    return P2_list

#  triangulation
# will be called by Triangulation
# return sind
def triangulation(P1, P2, x1, x2):
    """
    x1[0] /= x1[2]
    x1[1] /= x1[2]
    x2[0] /= x2[2]
    x2[1] /= x2[2]
    """
    A = np.zeros((4, 4))
    p1_1 = P1[0, :]
    p1_2 = P1[1, :]
    p1_3 = P1[2, :]
    
    p2_1 = P2[0, :]
    p2_2 = P2[1, :]
    p2_3 = P2[2, :]
    
    A[0,:] = x1[0] * p1_3 - p1_1
    A[1,:] = x1[1] * p1_3 - p1_2
    A[2,:] = x2[0] * p2_3 - p2_1
    A[3,:] = x2[1] * p2_3 - p2_2
    
    U, S, Vt = np.linalg.svd(A)
    #Vt=Vt.T
    x = Vt[:, -1]
    x[0] /= x[3]
    x[1] /= x[3]
    x[2] /= x[3]
    return x[:3]

#  triangulation, will call triangulation
def Triangulation(P1, P2, X1, X2):
    n = X1.shape[0]
    X = []
    for i in range(n):
        x = triangulation(P1, P2, X1[i], X2[i])
        X.append(x)
    return np.asarray(X)

# plot 3d points
def plot_3d(pts):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for pt in pts:
        ax.scatter(pt[0], pt[1], pt[2], c = 'b')
    
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()
    
img1 = cv2.imread('./Statue1.bmp')
img2 = cv2.imread('./Statue2.bmp')
img2_original = cv2.imread('./Statue2.bmp')

gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

matches, match_index_img1, match_index_img2, good_kp1, good_kp2 = get_co_points(gray1, gray2)

match_index_img1_norm, T1 = image_co_normalize(match_index_img1)
match_index_img2_norm, T2 = image_co_normalize(match_index_img2)

F = get_F(match_index_img1_norm, match_index_img2_norm, T1, T2)
#optimal_F = get_F_RANSAC(match_index_img1_norm, match_index_img2_norm,match_index_img1, match_index_img2, T1, T2)

r = random.randint(0,255)
g = random.randint(0,255)
b = random.randint(0,255)

#cv2.drawKeypoints(img2 , good_kp2, img2_points, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.drawKeypoints(img1, good_kp1, img1, color=(255,0,0), flags=0)
plt.imshow(img2),plt.show()

print(F)
img2_epipolar = draw_epipolar(match_index_img1, match_index_img2, F, img2_original)
match_image = cv2.hconcat([img1, img2_epipolar])
plt.imshow(match_image),plt.show()

K1 = np.array([[5426.566895, 0.678017, 330.096680], 
                         [0, 5423.133301, 648.950012],
                         [0, 0, 1]])
K2 = np.array([[5426.566895, 0.678017, 387.430023], 
                         [0, 5423.133301, 620.616699],
                         [0, 0, 1]])

P2_list = []
P2_list = P2(K1, K2, F)

"""
X1 = []
X2 = []
match_index = match
for pts in match_index:
    X1.append(pts[0])
    X2.append(pts[1])
X1 = np.asarray(X1, dtype = 'float32')
X2 = np.asarray(X2, dtype = 'float32')
"""
X1 = match_index_img1
X2 = match_index_img2
X1 = np.asarray(X1, dtype = 'float32')
X2 = np.asarray(X2, dtype = 'float32')

#pts_3d = Triangulation(P1, P2_list[3], X1, X2)

P1 = np.array([[1, 0, 0, 0], 
                         [0, 1, 0, 0],
                         [0, 0, 1, 0]])
for P2 in P2_list:  
    pts_3d = Triangulation(P1, P2, X1, X2)
    plot_3d(pts_3d)