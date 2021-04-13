# coding: utf-8
"""Train VC model.

"""

from datetime import datetime
import glob
import os

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

import config
from data import SpeechGenDataset
from model import SpeechRecogModel, SpeechGenModel, DiscrimintorModel

device = "cuda:{}".format(config.GPU_ID) if torch.cuda.is_available() else "cpu"

def train(train_loader, recog_model, gen_model, dis_model, criterion_G, optimizer_G, optimizer_D, scheduler_G, scheduler_D):
    running_loss_G = 0.0
    running_loss_D = 0.0
    running_loss_MSE = 0.0

    for x, y in tqdm(train_loader, total=len(train_loader), desc='train'):
        x, y = x.to(device), y.to(device)

        # Train Discriminator
        gen_model.eval()
        dis_model.train()

        optimizer_D.zero_grad()

        output_real = dis_model(y)

        x_tmp = recog_model(x)
        x_tmp = F.softmax(x_tmp, dim=1)
        x_tmp = gen_model(x_tmp)
        output_fake = dis_model(x_tmp)

        loss_D = - torch.mean(output_real) + torch.mean(output_fake)

        loss_D.backward()
        optimizer_D.step()

        running_loss_D += loss_D.item()

        # Clip weights of Discriminator
        for p in dis_model.parameters():
            p.data.clamp_(-0.01, 0.01)

        # Train Generator
        gen_model.train()
        dis_model.eval()

        optimizer_G.zero_grad()

        x_tmp = recog_model(x)
        x_tmp = F.softmax(x_tmp, dim=1)
        x_tmp = gen_model(x_tmp)
        output_fake = dis_model(x_tmp)

        loss_G = - torch.mean(output_fake) * 5
        loss_MSE = criterion_G(x_tmp, y) 
        loss_G_all = loss_G + loss_MSE

        loss_G_all.backward()
        optimizer_G.step()

        running_loss_G += loss_G.item()
        running_loss_MSE += loss_MSE.item()

    train_loss_D = running_loss_D / len(train_loader)
    train_loss_G = running_loss_G / len(train_loader)
    train_loss_MSE = running_loss_MSE / len(train_loader)

    # scheduler.step()

    return train_loss_D, train_loss_G, train_loss_MSE

def valid(valid_loader, recog_model, gen_model, dis_model, criterion_G):
    gen_model.eval()
    dis_model.eval()

    running_loss_G = 0.0
    running_loss_D = 0.0
    running_loss_MSE = 0.0

    with torch.no_grad():
        for x, y in tqdm(valid_loader, total=len(valid_loader), desc='valid'):
            x, y = x.to(device), y.to(device)

            output_real = dis_model(y)

            x_tmp = recog_model(x)
            x_tmp = F.softmax(x_tmp, dim=1)
            x_tmp = gen_model(x_tmp)
            output_fake = dis_model(x_tmp)

            loss_D = - torch.mean(output_real) + torch.mean(output_fake)

            running_loss_D += loss_D.item()

            x_tmp = recog_model(x)
            x_tmp = F.softmax(x_tmp, dim=1)
            x_tmp = gen_model(x_tmp)
            output_fake = dis_model(x_tmp)

            loss_G = - torch.mean(output_fake) * 5
            loss_MSE = criterion_G(x_tmp, y)
            loss_G_all = loss_G + loss_MSE

            running_loss_G += loss_G.item()
            running_loss_MSE += loss_MSE.item()

        valid_loss_G = running_loss_G / len(valid_loader)
        valid_loss_D = running_loss_D / len(valid_loader)
        valid_loss_MSE = running_loss_MSE / len(valid_loader)

    return valid_loss_D, valid_loss_G, valid_loss_MSE


def main():
    ct = datetime.now()
    log_dir = os.path.join(config.LOG_ROOT, "VC", ct.strftime("%Y%m%d_%H%M%S"))

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    else:
        raise FileExistsError("Directory {} already exists".format(log_dir))

    print("Output dir: {}".format(log_dir))

    train_loader = torch.utils.data.DataLoader(
        SpeechGenDataset(phase = "train", data = "train", spk = config.VC_SPEAKER),
        batch_size = config.VC_BATCH_SIZE,
        num_workers = config.NUM_WORKERS,
        shuffle = True
    )

    valid_loader = torch.utils.data.DataLoader(
        SpeechGenDataset(phase = "train", data = "valid", spk = config.VC_SPEAKER),
        batch_size = config.VC_BATCH_SIZE,
        num_workers = config.NUM_WORKERS,
        shuffle = True
    )
    
    recog_model_path = glob.glob(os.path.join(config.ASR_INFERENCE_ROOT, "*best.pth"))[0]

    recog_model = SpeechRecogModel().to(device)
    recog_model.load_state_dict(torch.load(recog_model_path, map_location='cuda:0'))
    recog_model.eval()

    gen_model = SpeechGenModel().to(device)
    dis_model = DiscrimintorModel().to(device)
    print(recog_model)
    print(gen_model)
    print(dis_model)

    optimizer_G = optim.Adam(
        gen_model.parameters(),
        lr = config.VC_LR
    )

    optimizer_D = optim.Adam(
        dis_model.parameters(),
        lr = config.VC_LR
    )

    scheduler_G = StepLR(optimizer_G, step_size=100, gamma=0.1)
    scheduler_D = StepLR(optimizer_D, step_size=100, gamma=0.1)

    criterion_G = nn.MSELoss()

    best_loss = None
    best_model = None

    for epoch in range(1, config.VC_N_EPOCHS + 1):
        train_loss_D, train_loss_G, train_loss_MSE = train(train_loader, recog_model, gen_model, dis_model, criterion_G, optimizer_G, optimizer_D, scheduler_G, scheduler_D)
        valid_loss_D, valid_loss_G, valid_loss_MSE = valid(valid_loader, recog_model, gen_model, dis_model, criterion_G)

        print("epoch [{}/{}]\ntrain_loss_D: {:.3f}, train_loss_G: {:.3f}, train_loss_MSE: {:.3f},\nvalid_loss_D: {:.3f}, valid_loss_G: {:.3f}, valid_loss_MSE: {:.3f}"\
                .format(epoch, config.VC_N_EPOCHS, train_loss_D, train_loss_G, train_loss_MSE, valid_loss_D, valid_loss_G, valid_loss_MSE))

        # if epoch == 1 or (valid_loss < best_loss):
        #     if best_loss is not None:
        #         print('  => valid_loss improved from {:.3f} to {:.3f}!'.format(best_loss, valid_loss))
        #         os.remove(best_model)

        #     best_loss = valid_loss
        #     best_model = os.path.join(log_dir,  'epoch{:03d}_{:.3f}_best.pth'.format(epoch, valid_loss))
        #     torch.save(gen_model.state_dict(), best_model)
        if epoch % config.SAVE_MODEL_FREQ == 0:
            current_model = os.path.join(log_dir, 'epoch{:03d}_{:.3f}.pth'.format(epoch, valid_loss_G))
            torch.save(gen_model.state_dict(), current_model)

if __name__ == "__main__":
    main()
