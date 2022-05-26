import matplotlib.pyplot as plt
import torch

import config


def plot_D_pridiction_iteration(num_epoch,start_epoch = 0,iteration = 270):
    DR = []
    DF = []
    for epoch in range(start_epoch, num_epoch):
        curDR = torch.load(config.DISC_VALUE_DIR+f"real_{epoch}")
        curDF = torch.load(config.DISC_VALUE_DIR+f"fake_{epoch}")
        curDR = list(curDR)
        curDF = list(curDF)
        DR += curDR
        DF += curDF

    x = range(start_epoch * iteration, (num_epoch - start_epoch) * iteration + start_epoch * iteration)
    plt.plot(x, DR, color="green", label="DR")
    plt.plot(x, DF, color="red", label="DF")
    plt.legend()
    plt_title = 'BATCH_SIZE = 16; LEARNING_RATE:2e-4'
    plt.title(plt_title)
    plt.xlabel('Iteration')
    plt.ylabel('Disc_pridiction')
    # plt.savefig(file_name)
    plt.show()

def plot_D_pridiction(num_epoch,start_epoch = 0):
    DR = []
    DF = []
    for epoch in range(start_epoch, num_epoch):
        curDR = torch.load(config.DISC_VALUE_DIR+f"real_{epoch}")
        curDF = torch.load(config.DISC_VALUE_DIR+f"fake_{epoch}")
        DR.append(curDR.mean().item())
        DF.append(curDF.mean().item())

    x = range(start_epoch, num_epoch)
    plt.plot(x, DR, color="green", label="DR")
    plt.plot(x, DF, color="red", label="DF")
    plt.legend()
    plt_title = 'BATCH_SIZE = 16; LEARNING_RATE:2e-4'
    plt.title(plt_title)
    plt.xlabel('Epoch')
    plt.ylabel('Disc_pridiction')
    # plt.savefig(file_name)
    plt.show()

def plot_loss_iteration(num_epoch,start_epoch = 0,iteration = 270):
    y_D = []
    y_G = []
    for epoch in range (start_epoch,num_epoch):
        curDLoss = torch.load(config.LOSS_DIR+f"D_{epoch}")
        curGLoss = torch.load(config.LOSS_DIR+f"G_{epoch}")
        curDLoss = list(curDLoss)
        curGLoss = list(curGLoss)
        y_D += curDLoss
        y_G += curGLoss

    x = range(start_epoch*iteration,(num_epoch-start_epoch)*iteration+start_epoch*iteration)
    plt.plot(x, y_D, color="green",label="D_loss")
    plt.plot(x, y_G, color="red",label="G_loss")
    plt.legend()
    plt_title = 'BATCH_SIZE = 16; LEARNING_RATE:2e-4'
    plt.title(plt_title)
    plt.xlabel('Iteration')
    plt.ylabel('LOSS')
    # plt.savefig(file_name)
    plt.show()

def plot_loss(num_epoch,start_epoch = 0):
    y_D = []
    y_G = []
    for epoch in range (start_epoch,num_epoch):
        curDLoss = torch.load(config.LOSS_DIR+f"D_{epoch}")
        curGLoss = torch.load(config.LOSS_DIR+f"G_{epoch}")
        # curDLoss = list(curDLoss)
        # curGLoss = list(curGLoss)
        y_D.append(curDLoss.mean().item())
        y_G.append(curGLoss.mean().item())

    x = range(start_epoch,num_epoch)
    plt.plot(x, y_D, color="green",label="D_loss")
    plt.plot(x, y_G, color="red",label="G_loss")
    plt.legend()
    plt_title = 'BATCH_SIZE = 16; LEARNING_RATE:2e-4'
    plt.title(plt_title)
    plt.xlabel('Epoch')
    plt.ylabel('LOSS')
    # plt.savefig(file_name)
    plt.show()

if __name__ == "__main__":
    # plot_loss_iteration(76)
    # plot_D_pridiction_iteration(3)
    plot_loss(500)
    plot_D_pridiction(500)
