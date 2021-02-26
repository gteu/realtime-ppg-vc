# coding: utf-8
"""Train ASR model.

"""

from datetime import datetime
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import config
from data import SpeechFeatureDataset
from model import SpeechRecogModel

device = "cuda:{}".format(config.GPU_ID) if torch.cuda.is_available() else "cpu"

def train(train_loader, model, criterion, optimizer):
    model.train()

    running_loss = 0.0
    running_acc = 0.0

    for x, y in tqdm(train_loader, total=len(train_loader), desc='train'):
        x, y = x.to(device), y.to(device)

        output = model(x)

        loss = criterion(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        output = output.data.cpu().numpy()
        output = np.argmax(output, axis=1)
        y = y.data.cpu().numpy()
        acc = np.mean((output == y).astype(int))

        running_acc += acc

    train_loss = running_loss / len(train_loader)
    train_acc = running_acc / len(train_loader)

    return train_loss, train_acc * 100

def valid(valid_loader, model, criterion):
    model.eval()

    running_loss = 0.0
    running_acc = 0.0

    with torch.no_grad():
        for x, y in tqdm(valid_loader, total=len(valid_loader), desc='valid'):
            x, y = x.to(device), y.to(device)

            output = model(x)

            loss = criterion(output, y)

            running_loss += loss.item()

            output = output.data.cpu().numpy()
            output = np.argmax(output, axis=1)
            y = y.data.cpu().numpy()
            acc = np.mean((output == y).astype(int))

            running_acc += acc

        valid_loss = running_loss / len(valid_loader)
        valid_acc = running_acc / len(valid_loader)

    return valid_loss, valid_acc * 100


def main():
    ct = datetime.now()
    log_dir = os.path.join(config.LOG_ROOT, ct.strftime("%Y%m%d_%H%M%S"))

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    else:
        raise FileExistsError("Directory {} already exists".format(log_dir))

    print("Output dir: {}".format(log_dir))

    train_loader = torch.utils.data.DataLoader(
        SpeechFeatureDataset(phase = "train", data = "train"),
        batch_size = config.BATCH_SIZE,
        num_workers = config.NUM_WORKERS,
        shuffle = True
    )

    valid_loader = torch.utils.data.DataLoader(
        SpeechFeatureDataset(phase = "train", data = "valid"),
        batch_size = config.BATCH_SIZE,
        num_workers = config.NUM_WORKERS,
        shuffle = True
    )

    model = SpeechRecogModel().to(device)
    print(model)

    optimizer = optim.Adam(
        model.parameters(),
        lr = config.LR
    )

    criterion = nn.CrossEntropyLoss()

    best_loss = None
    best_model = None

    for epoch in range(1, config.N_EPOCHS + 1):
        train_loss, train_acc = train(train_loader, model, criterion, optimizer)
        valid_loss, valid_acc = valid(valid_loader, model, criterion)

        print("epoch [{}/{}] train_loss: {:.3f}, train_acc: {:.3f}%, valid_loss: {:.3f}, valid_acc: {:.3f}%"\
                .format(epoch, config.N_EPOCHS, train_loss, train_acc, valid_loss, valid_acc))

        if epoch == 1 or (valid_loss < best_loss):
            if best_loss is not None:
                print('  => valid_loss improved from {:.3f} to {:.3f}!'.format(best_loss, valid_loss))
                os.remove(best_model)

            best_loss = valid_loss
            best_model = os.path.join(log_dir,  'epoch{:03d}_{:.3f}_best.pth'.format(epoch, valid_loss))
            torch.save(model.state_dict(), best_model)
        if epoch % config.SAVE_MODEL_FREQ == 0:
            current_model = os.path.join(log_dir, 'epoch{:03d}_{:.3f}.pth'.format(epoch, valid_loss))
            torch.save(model.state_dict(), current_model)


if __name__ == "__main__":
    main()
