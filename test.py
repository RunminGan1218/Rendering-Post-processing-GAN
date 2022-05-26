import torch
import config
from utils import load_model
from discriminator_model import Discriminator
from generator_model import Generator
from dataset import MyDataset
from torch.utils.data import DataLoader
import numpy as np
from evaluation_methods import ERROR_calculator
from tqdm import tqdm
from PIL import Image

def test(gen, loader, save_img = False):
    loop = tqdm(loader, leave=True)
    RMSE = 0
    SSIM = 0
    BER = 0
    ACC = 0
    it_num = 0

    gen.eval()
    for x, y, label in loop:
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)
        with torch.no_grad():
            y_fake = gen(x)
            # print(y_fake.shape)
            y_fake = (y_fake*0.5 + 0.5).cpu().squeeze().mul(255).add_(0.5).clamp(0,255).permute(1,2,0).type(torch.uint8).numpy()
            y = (y*0.5 + 0.5).cpu().squeeze().mul(255).add_(0.5).clamp(0,255).permute(1,2,0).type(torch.uint8).numpy()
            if save_img:
                image = np.hstack((y_fake, y))
                image = Image.fromarray(image)
                image.save(f"test_result/7/Ladv+L1/{it_num}.jpg")
            calculator = ERROR_calculator(y_fake, y, 3)
            RMSE += calculator.RMSE()
            SSIM += calculator.SSIM()
            BER += calculator.BER()
            ACC += calculator.ACC()
            it_num += 1

    print("RMSE:{}".format(RMSE / it_num))
    print("SSIM:{}".format(SSIM / it_num))
    print(f"BER:{BER / it_num}")
    print(f"ACC:{ACC / it_num}")

def Disc_Calculator(gen,disc,loader):
    loop = tqdm(loader, leave=True)
    it_num = 0
    D_real_sum = 0
    D_fake_sum = 0
    for idx,(x, y, label) in enumerate(loop):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)
        gen.eval()
        disc.eval()
        with torch.no_grad():
            y_fake = gen(x)
            D_real = disc(x,y)
            D_fake = disc(x,y_fake)
            D_real = torch.sigmoid(D_real).mean().item()
            D_fake = torch.sigmoid(D_fake).mean().item()
            D_real_sum += D_real
            D_fake_sum += D_fake

        if idx % 1 == 0:
            loop.set_postfix(
                D_real = D_real,
                D_fake = D_fake,
            )
        it_num += 1

    print(D_real_sum / it_num)
    print(D_fake_sum / it_num)

def time_measuring(gen,loader,):
    starter, ender = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    timearray = []
    gen.eval()
    for idx, (x, y, label) in enumerate(loader):
        x = x.to(config.DEVICE)
        with torch.no_grad():
            starter.record()
            y_fake = gen(x)
            ender.record()
            torch.cuda.synchronize()
            time_interval = starter.elapsed_time(ender)
            if (idx >= 1):
                timearray.append(time_interval)
    # timearray = torch.tensor(timearray)
    return sum(timearray)/len(timearray)
    # return timearray

def main(model_epoch):
    disc = Discriminator(in_channels_x=4,in_channels=3).to(config.DEVICE)
    gen = Generator(in_channels=4,features=64).to(config.DEVICE)


    load_model(
        config.MOD_DIR + f"{model_epoch}-" + config.CHECKPOINT_GEN, gen
    )
    load_model(
        config.MOD_DIR + f"{model_epoch}-" + config.CHECKPOINT_DISC, disc
    )

    val_dataset = MyDataset(config.TEST_DIR)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=1,num_workers=config.NUM_WORKERS)
    val_dataloader_ = DataLoader(dataset=val_dataset, batch_size=32, num_workers=config.NUM_WORKERS,drop_last=True)

    test(gen, val_dataloader, True)
    # Disc_Calculator(gen, disc, val_dataloader_)
    # print(time_measuring(gen, val_dataloader))    #测试时间时使用


if __name__ == "__main__":
    main(495)