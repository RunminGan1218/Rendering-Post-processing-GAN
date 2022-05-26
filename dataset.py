from cv2 import normalize
import numpy as np
import torch
import config
import os
import lightpic_gen
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
# from torchvision.transforms import ToTensor,Normalize,Compose


class MyDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        image = np.array(Image.open(img_path))
        input_image = image[:, :1024, :]
        target_image = image[:, 1024:, :]


        
        #添加光照层图片
        light_label = int(img_file[-7:-4])
        # trans = Compose([ToTensor(),Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        # tensorTrans = ToTensor()
        # normalize = Normalize(mean=[128], std=[128])
        light_pic =lightpic_gen.draw_lightpic(512,512,light_label)   


        augmentations = config.both_transform(image=input_image, image0=target_image)
        input_image = augmentations["image"]
        target_image = augmentations["image0"]

        input_image = config.transform_only_input(image=input_image)["image"]
        target_image = config.transform_only_mask(image=target_image)["image"]

        # print(input_image.shape)
        # print(light_pic.shape)

        input_image = torch.cat([input_image,light_pic],dim=0).float()

        return input_image, target_image, light_label

# train_dataset = MapDataset(root_dir=config.TRAIN_DIR)
# input,output,label =  train_dataset[0]
# print(input.dtype)

if __name__ == "__main__":
    dataset = MyDataset("data/train/")
    loader = DataLoader(dataset, batch_size=5)
    for x, y, label in loader:
        print(x.shape)
        save_image(x, "x.png")
        save_image(y, "y.png")
        import sys

        sys.exit()
