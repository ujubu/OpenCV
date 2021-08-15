import cv2
import numpy as np

#book cover that downloaded from net
img1 = cv2.imread("vakif.jpg", 0)

#my book cover
img2 = cv2.imread("myVakif.jpeg", 0)

#resizing images for better seeing
w1, h1 = int(img1.shape[1]*0.3), int(img1.shape[0]*0.3)
dim1 = (w1, h1)
w2, h2 = int(img2.shape[1]*0.6), int(img2.shape[0]*0.6)
dim2 = (w2, h2)

img1 = cv2.resize(img1, dim1, interpolation = cv2.INTER_AREA)
img2 = cv2.resize(img2, dim2, interpolation = cv2.INTER_AREA)

#keypoints detecting
orb = cv2.ORB_create(nfeatures = 1000)

kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

imgKp1 = cv2.drawKeypoints(img1, kp1, None)
imgKp2 = cv2.drawKeypoints(img2, kp2, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2) #Vakıf için

goodMatch = []
for k,l in matches:
    if k.distance < 0.75*l.distance:
        goodMatch.append([k])
        
print(len(goodMatch))
        
imgDetect = cv2.drawMatchesKnn(img1, kp1, img2, kp2, goodMatch, None, flags=2)

# #image showing
# cv2.imshow("img1", img1)
# cv2.imshow("img2", img2)

# #featured image showing
# cv2.imshow("imgKp1", imgKp1)
# cv2.imshow("imgKp2", imgKp2)

cv2.imshow("imgDetect", imgDetect)

cv2.waitKey(0)


'''
a code by
        _       _           
       (_)     | |          
 _   _  _ _   _| | _  _   _ 
| | | || | | | | || \| | | |
| |_| || | |_| | |_) ) |_| |
 \____|| |\____|____/ \____| 
     (__/                              
 
'''
