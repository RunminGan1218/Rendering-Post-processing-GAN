
import torch
from torch import nn
from torchvision import models
# import numpy as np
# from turtle import forward
# from torch.autograd import Variable
import torchvision.transforms as transforms

class MyVGG16(nn.Module):
    def __init__(self):
        super().__init__()
        m = models.vgg16(pretrained=True)
        self.model = nn.Sequential(*list(m.features)[:24])
        self.model.eval()
    
    def forward(self,x):
        x = self.image_trans(x)
        # print(x.shape)
        return self.model(x)

    def image_trans(self,Data):
        trans = transforms.Resize((224,224))
        return trans(Data)


# class CNNShow():
#     def __init__(self, model):
#         self.model = model
#         self.model.eval()
 
#         self.created_image = self.image_for_pytorch(np.uint8(np.random.uniform(150, 180, (224, 224, 3))))
 
 
#     def show(self):
#         x = self.created_image
#         for index, layer in enumerate(self.model):
#             print(index,layer)
#             x = layer(x)
 
#     def image_for_pytorch(self,Data):
#         transform = transforms.Compose([
#             transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
#             transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
#         ]
#         )
#         imData = transform(Data)
#         imData = Variable(torch.unsqueeze(imData, dim=0), requires_grad=True)
#         return imData
 
# if __name__ == '__main__':
 
#     pretrained_model = models.vgg16(pretrained=True).features
#     CNN = CNNShow(pretrained_model)
#     CNN.show()

if __name__ == '__main__':
    vgg16 = MyVGG16()
    # print(vgg16)
    x = torch.randn((16, 3, 256, 256))
    feature = vgg16(x)
    print(feature.shape)