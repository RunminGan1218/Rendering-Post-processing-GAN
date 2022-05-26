import numpy as np
from PIL import Image
import cv2
import albumentations as A

class ERROR_calculator:

    #请传入np数组
    def __init__(self, fake, real, channel=3):
        self.fake = fake
        self.real = real
        self.channel = channel


    def RGB2GRAY(self):
        gray_real = cv2.cvtColor(self.real, cv2.COLOR_RGB2GRAY)
        gray_fake = cv2.cvtColor(self.fake, cv2.COLOR_RGB2GRAY)
        return gray_real,gray_fake
    
    def shadow_extract(self, temp_real, temp_fake):
        temp_real[temp_real < 120] = 0
        temp_real[temp_real >= 240] = 0
        temp_real[temp_real != 0] =255

        temp_fake[temp_fake < 120] = 0
        temp_fake[temp_fake >= 240] = 0
        temp_fake[temp_fake != 0] =255

    def RMSE(self):
        def single_RMSE(pic1, pic2):
            pic1 = pic1.astype(np.float32)
            pic2 = pic2.astype(np.float32)
            diff = pic1 - pic2
            # print(diff)
            mse = np.mean(np.square(diff))
            return np.sqrt(mse)
        if self.channel != 1:
            rmse_sum = 0

            for band in range(self.channel):
                fake_band = self.fake[:,:,band]
                real_band = self.real[:,:,band]
                rmse_sum += single_RMSE(fake_band,real_band)
            
            return rmse_sum
        else:
            return single_RMSE(self.fake,self.real)


    def SSIM(self):
        def ssim(img1, img2, L = 255):
            K1 = 0.01
            K2 = 0.03
            C1 = (K1 * L)**2
            C2 = (K2 * L)**2
            C3 = C2/2

            img1 = img1.astype(np.float32)
            img2 = img2.astype(np.float32)
            # ux
            ux = img1.mean()
            # uy
            uy = img2.mean()
            # ux^2
            ux_sq = ux**2
            # uy^2
            uy_sq = uy**2
            # ux*uy
            uxuy = ux * uy
            # ox、oy方差计算
            ox_sq = img1.var()
            oy_sq = img2.var()
            ox = np.sqrt(ox_sq)
            oy = np.sqrt(oy_sq)
            oxoy = ox * oy
            oxy = np.mean((img1 - ux) * (img2 - uy))
            # 公式一计算
            L = (2 * uxuy + C1) / (ux_sq + uy_sq + C1)
            C = (2 * ox * oy + C2) / (ox_sq + oy_sq + C2)
            S = (oxy + C3) / (oxoy + C3)
            ssim = L * C * S
            # 验证结果输出
            # print('ssim:', ssim, ",L:", L, ",C:", C, ",S:", S)
            return ssim

        if self.channel != 1:
            ssim_sum = 0
            for band in range(self.channel):
                fake_band = self.fake[:,:,band]
                real_band = self.real[:,:,band]
                ssim_sum += ssim(fake_band,real_band)
            return ssim_sum/self.channel

        else:
            return ssim(self.fake,self.real)


    def BER(self):
        gray_real,gray_fake = self.RGB2GRAY()
        self.shadow_extract(gray_real,gray_fake)
        # img1 = Image.fromarray(gray_real)
        # img2 = Image.fromarray(gray_fake)
        # img1.show()
        # img2.show()
        TP_ = gray_real&gray_fake  #1
        TN_ = gray_real|gray_fake  #0
        FP_ = TP_^gray_fake        #1
        FN_ = TP_^gray_real        #1
        TP = np.sum(TP_==255)
        TN = np.sum(TN_==0)
        FP = np.sum(FP_==255)
        FN = np.sum(FN_==255)
        FPR = FP/(TN+FP)
        FNR = FN/(TP+FN)
        return (FPR+FNR)/2
         

    def ACC(self):
        gray_real,gray_fake = self.RGB2GRAY()
        self.shadow_extract(gray_real,gray_fake)
        TP_ = gray_real&gray_fake   #1
        TN_ = gray_real|gray_fake   #0
        TP = np.sum(TP_==255)
        TN = np.sum(TN_==0)
        return (TP+TN)/(256*256)
    


if __name__ == "__main__":
    realimg_path = "evaluation2/label_499_137.png"
    fakeimg_path = "evaluation2/y_gen_499_137.png"
    # fakeimg_path = "evaluation1/y_gen_285_215.png"

    realimg = cv2.imread(realimg_path)
    realimg = realimg[:, :, ::-1]
    # resize = A.Resize(256,256)
    # realimg = resize(image = realimg)["image"]
    fakeimg = cv2.imread(fakeimg_path)
    print(realimg.shape)
    fakeimg = fakeimg[:, :, ::-1]

    pic_ERROR = ERROR_calculator(fakeimg,realimg,3)
    print(pic_ERROR.RMSE())
    print(pic_ERROR.SSIM())
    print(pic_ERROR.BER())
    print(pic_ERROR.ACC())

    

    # gray_real = cv2.cvtColor(realimg, cv2.COLOR_RGB2GRAY)
    # gray_fake = cv2.cvtColor(fakeimg, cv2.COLOR_RGB2GRAY)
    


    # # img1 = Image.fromarray(gray_real)
    # # img2 = Image.fromarray(gray_fake)
    # # img1.show()
    # # img2.show()
    

    # temp_real = gray_real

    # temp_fake = gray_fake


    # temp_real[temp_real < 130] = 0
    # temp_real[temp_real >= 240] = 0
    # temp_real[temp_real != 0]  =255

    # temp_fake[temp_fake < 130] = 0
    # temp_fake[temp_fake >= 240] = 0
    # temp_fake[temp_fake != 0]  =255

    # TP = temp_real&temp_fake  #1
    # TN = temp_real|temp_fake  #0
    # FP = TP^temp_fake         #1
    # FN = TP^temp_real         #1
    # # print(np.max(TP))

    # img1 = Image.fromarray(TP)
    # img2 = Image.fromarray(TN)
    # img3 = Image.fromarray(FP)
    # img4 = Image.fromarray(FN)
    # img1.show()
    # img2.show()
    # img3.show()
    # img4.show()