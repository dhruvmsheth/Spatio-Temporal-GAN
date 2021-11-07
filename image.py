import cv2
import numpy as np
import cmapy
import matplotlib.pyplot as plt 


img =  cv2.imread('/home/pi/opencv/video0007_frame0007gt_R128x128.png').astype(np.float)  # BGR, float

blue = img[:,:,2]
green = img[:,:,1]
red = img[:,:,0]
exg = 2*green - red - blue
print("max exg", exg.max())
print("mean exg", exg.mean())
print("min exg", exg.min())


img = np.where(exg < 0, 0, exg).astype('uint8')

exr = 1.4*red - green
exr = np.where(exr < 0, 0, exr).astype('uint8')

exgr = exg - exr
print("max exgr", exgr.max())
print("mean exgr", exgr.mean())
print("min exgr", exgr.min())
exgr = np.where(exgr < 25, 0, exgr).astype('uint8')


img = img.astype(np.uint8)  # convert back to uint8
exgr = exgr.astype(np.uint8)  # convert back to uint8
exr = exr.astype(np.uint8)  # convert back to uint8

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
out = clahe.apply(exgr)
im_color = cv2.applyColorMap(out, cv2.COLORMAP_INFERNO)
img_c = cv2.applyColorMap(exgr, cmapy.cmap('Reds')).astype(np.int)
_, R, NIR = cv2.split(img_c)

img_c = img_c.astype(np.uint8)  # convert back to uint8
img_c = cv2.applyColorMap(img_c, cv2.COLORMAP_INFERNO)



#cv2.imwrite('new-image.png', exgr)  # save the image
cv2.imshow('exr', exr)

cv2.imshow('img', img)
cv2.imshow('exgr', exgr)
cv2.imshow("colormap", im_color)
cv2.imshow("img_c", img_c)



cv2.waitKey()
