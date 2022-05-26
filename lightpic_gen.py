import numpy as np
# from PIL import Image
from torchvision.transforms import ToTensor

def draw_lightpic(w,h,angle):
    i = np.repeat(np.arange(h),w).reshape(h,w)-h//2
    j =  np.tile(np.arange(w), (h,1))-w//2
    d = np.hypot(i,j)
    d = np.int32(d)
    # d = np.uint8(255*(d-d.min())/(d.max()-d.min()))
    d = 255-d
    # Image.fromarray(d).show()

    if angle<45 or angle >=315:
        angle = (angle+45)%360
        start = round((angle/90)*255)
        pic = d[0:256,start:start+256]
    elif angle >=45 and angle <135:
        angle = angle-45
        start = round((angle/90)*255)
        pic = d[start:start+256,256:]
    elif angle >=135 and angle < 225:
        angle = angle-135
        end = round((angle/90)*255)
        pic = d[256:,256-end:512-end]
    elif angle >=225 and angle <315:
        angle = angle - 225
        end = round((angle/90)*255)
        pic = d[256-end:512-end,0:256]

    pic[pic<0] = 0
    # Image.fromarray(pic).show()
    # print(pic)
    pic = (pic-128)/128
    tensortrans = ToTensor()
    pic = tensortrans(pic)
    # print(pic)
    

    return pic


# draw_lightpic(512,512,0)