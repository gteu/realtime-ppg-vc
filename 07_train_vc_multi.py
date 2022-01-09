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
from data import SpeechGenMultiDataset
from model import SpeechRecogModel, SpeechGenMultiModel

device = "cuda:{}".format(config.GPU_ID) if torch.cuda.is_available() else "cpu"

def train(train_loader, recog_model, gen_model, criterion, optimizer, scheduler):
    gen_model.train()

    running_loss = 0.0

    for x, y, spk in tqdm(train_loader, total=len(train_loader), desc='train'):
        x, y, spk = x.to(device), y.to(device), spk.to(device)

        output = recog_model(x)
        output = F.softmax(output, dim=1)
        output = gen_model(output, spk)

        loss = criterion(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)

    scheduler.step()

    return train_loss

def valid(valid_loader, recog_model, gen_model, criterion):
    gen_model.eval()

    running_loss = 0.0

    with torch.no_grad():
        for x, y, spk in tqdm(valid_loader, total=len(valid_loader), desc='valid'):
            x, y, spk = x.to(device), y.to(device), spk.to(device)

            output = recog_model(x)
            output = F.softmax(output, dim=1)
            output = gen_model(output, spk)

            loss = criterion(output, y)

            running_loss += loss.item()

        valid_loss = running_loss / len(valid_loader)

    return valid_loss


def main():
    ct = datetime.now()
    log_dir = os.path.join(config.LOG_ROOT, "VC", "multi_" + ct.strftime("%Y%m%d_%H%M%S"))

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    else:
        raise FileExistsError("Directory {} already exists".format(log_dir))

    print("Output dir: {}".format(log_dir))

    train_loader = torch.utils.data.DataLoader(
        SpeechGenMultiDataset(phase = "train", data = "train"),
        batch_size = config.VC_BATCH_SIZE,
        num_workers = config.NUM_WORKERS,
        shuffle = True
    )

    valid_loader = torch.utils.data.DataLoader(
        SpeechGenMultiDataset(phase = "train", data = "valid"),
        batch_size = config.VC_BATCH_SIZE,
        num_workers = config.NUM_WORKERS,
        shuffle = True
    )
    
    recog_model_path = glob.glob(os.path.join(config.ASR_INFERENCE_ROOT, "*best.pth"))[0]

    recog_model = SpeechRecogModel().to(device)
    recog_model.load_state_dict(torch.load(recog_model_path, map_location='cuda:0'))
    recog_model.eval()

    gen_model = SpeechGenMultiModel().to(device)
    print(recog_model)
    print(gen_model)

    optimizer = optim.Adam(
        gen_model.parameters(),
        lr = config.VC_LR
    )

    scheduler = StepLR(optimizer, step_size=500, gamma=0.1)

    criterion = nn.MSELoss()

    best_loss = None
    best_model = None

    for epoch in range(1, config.VC_N_EPOCHS + 1):
        train_loss = train(train_loader, recog_model, gen_model, criterion, optimizer, scheduler)
        valid_loss = valid(valid_loader, recog_model, gen_model, criterion)

        print("epoch [{}/{}] train_loss: {:.3f}, valid_loss: {:.3f}"\
                .format(epoch, config.VC_N_EPOCHS, train_loss, valid_loss))

        if epoch == 1 or (valid_loss < best_loss):
            if best_loss is not None:
                print('  => valid_loss improved from {:.3f} to {:.3f}!'.format(best_loss, valid_loss))
                os.remove(best_model)

            best_loss = valid_loss
            best_model = os.path.join(log_dir,  'epoch{:03d}_{:.3f}_best.pth'.format(epoch, valid_loss))
            torch.save(gen_model.state_dict(), best_model)
        if epoch % config.SAVE_MODEL_FREQ == 0:
            current_model = os.path.join(log_dir, 'epoch{:03d}_{:.3f}.pth'.format(epoch, valid_loss))
            torch.save(gen_model.state_dict(), current_model)

if __name__ == "__main__":
    main()
