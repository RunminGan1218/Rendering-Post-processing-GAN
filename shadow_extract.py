import imghdr
from pickletools import uint8
import cv2
import numpy as np
from PIL import Image

realimg_path = ""
fakeimg_path = "evaluation1/y_gen_495_275.png"
# fakeimg_path = "evaluation1/y_gen_285_215.png"

fakeimg = cv2.imread(fakeimg_path)

# temp0 = np.zeros((256,256,1),np.uint8)
# temp1 = np.zeros((256,256,1),np.uint8)
temp2 = np.zeros((256,256,1),np.uint8)
# temp0 = fakeimg[:,:,0]
# temp1 = fakeimg[:,:,1]
temp2 = fakeimg[:,:,0]

temp2[temp2 < 100] = 0
# temp2[90 <=temp2<= 140] = 255
temp2[temp2 >= 240] = 0
temp2[temp2 != 0]  =255

# img0 = Image.fromarray(temp0)
# img1 = Image.fromarray(temp1)
img2 = Image.fromarray(temp2)

# img0.show()
# img1.show()
img2.show()