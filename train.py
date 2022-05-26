import torch
from utils import save_checkpoint, load_checkpoint, save_some_examples
import torch.nn as nn
import torch.optim as optim
import config
from dataset import MyDataset
from generator_model import Generator
from discriminator_model import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
from vgg16 import MyVGG16   # Lper损失
from torchvision.utils import save_image

torch.backends.cudnn.benchmark = True


def train_fn(
    disc, gen, loader, opt_disc, opt_gen, l1_loss, bce, g_scaler, d_scaler, epoch, vgg, mse_loss # Lper损失
):
    loop = tqdm(loader, leave=True)
    DLOSS = []
    GLOSS = []
    DR = []
    DF = []

    for idx, (x, y, label) in enumerate(loop):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)



        # Train Discriminator
        with torch.cuda.amp.autocast():        #上下文环境，主要是用于scaler，灵活缩放数据类型，用于减少所需显存
            y_fake = gen(x)
            D_real = disc(x, y)   # D_real有batch_size个层
            D_real_loss = bce(D_real, torch.ones_like(D_real))    # sizeaverage参数默认为true，对所有层的损失求平均取得一个标量损失值
            D_fake = disc(x, y_fake.detach())   # 分离，y_fake不求其中权重参数的梯度
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2  # 降低判别器的训练速度
            DLOSS.append(D_loss)

        disc.zero_grad()   # 每个batch，清空梯度空间，防止梯度累加
        d_scaler.scale(D_loss).backward()   #计算本次的梯度值（求loss的导数），放入梯度空间
        d_scaler.step(opt_disc)           #根据梯度空间中的梯度值，使用优化器对网络权重进行更新
        d_scaler.update()            #调整scaler中数据类型，以适应当前的网络

        # Train generator
        with torch.cuda.amp.autocast():
            D_fake = disc(x, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake)) * config.Ladv_LAMBDA

            # Lper损失
            V_f = vgg(y_fake)
            V_r = vgg(y)
            Lper = mse_loss(V_r,V_f) * config.Lper_LAMBDA

            L1 = l1_loss(y_fake, y) * config.L1_LAMBDA
            G_loss = G_fake_loss + L1 +Lper  # Lper损失
            GLOSS.append(G_loss)
            DR.append(torch.sigmoid(D_real).mean())
            DF.append(torch.sigmoid(D_fake).mean())


        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()



        # if idx % 10 == 0:   #每十个iteration计算一次当前平均的D_real和D_fake
        #     loop.set_postfix(     #在进度条后面加上后缀
        #         D_real=torch.sigmoid(D_real).mean().item(),    #求平均值，并转换为普通类型
        #         D_fake=torch.sigmoid(D_fake).mean().item(),
        #     )

    DLOSS = torch.tensor(DLOSS)
    GLOSS = torch.tensor(GLOSS)
    DR = torch.tensor(DR)
    DF = torch.tensor(DF)
    torch.save(DLOSS, config.LOSS_DIR+f"D_{epoch}")
    torch.save(GLOSS, config.LOSS_DIR+f"G_{epoch}")
    torch.save(DR, config.DISC_VALUE_DIR+f"real_{epoch}")
    torch.save(DF, config.DISC_VALUE_DIR+f"fake_{epoch}")

def main():
    disc = Discriminator(in_channels_x=4,in_channels=3).to(config.DEVICE)
    gen = Generator(in_channels=4, features=64).to(config.DEVICE)
    vgg = MyVGG16().to(config.DEVICE)   # Lper损失
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999),)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()
    MSE_LOSS = nn.MSELoss() # Lper损失
    start_epoch = 0

    if config.LOAD_MODEL:
        load_checkpoint(
            config.MOD_DIR+f"{config.START_EPOCH}-"+config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.MOD_DIR+f"{config.START_EPOCH}-"+config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE,
        )
        start_epoch = config.START_EPOCH + 1

    train_dataset = MyDataset(root_dir=config.TRAIN_DIR)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        drop_last=True,
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    val_dataset = MyDataset(root_dir=config.VAL_DIR)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    for epoch in range(start_epoch,config.NUM_EPOCHS):
        try:
            train_fn(
                disc, gen, train_loader, opt_disc, opt_gen, L1_LOSS, BCE, g_scaler, d_scaler, epoch, vgg, MSE_LOSS  # Lper损失
            )
        except RuntimeError as exception:
            if "out of memory" in str(exception):
                print("WARNING:CUDA out of memory!")
                if hasattr(torch.cuda, "empty_cache"):
                    torch.cuda.empty_cache()
            else:
                raise exception

        if config.SAVE_MODEL and epoch % 5 == 0:
            save_checkpoint(gen, opt_gen, filename=config.MOD_DIR+f"{epoch}-"+config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=config.MOD_DIR+f"{epoch}-"+config.CHECKPOINT_DISC)

        save_some_examples(gen, val_loader, epoch, folder="evaluation8")


if __name__ == "__main__":
    main()