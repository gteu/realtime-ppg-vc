# coding: utf-8

import os
import pickle
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn import functional as F

import config
from utils.scale import scale

class SpeechRecogDataset(Dataset):
    def __init__(self, phase, data):
        # load scaler
        scaler_file = os.path.join(config.ASR_DATA_DIR_PATH, "scaler.pkl")
        scaler = pickle.load(open(scaler_file, "rb"))
        self.X_mean, self.X_scale = scaler["X_mean"], scaler["X_scale"]

        # load files
        self.file_id_list = []
        file_id_list_path = os.path.join(config.ASR_DATA_DIR_PATH, data + ".list")
        with open(file_id_list_path) as fp:
            for line in fp:
                line = line.strip()
                self.file_id_list.append(line)

        self.x_data_path = os.path.join(config.ASR_DATA_DIR_PATH, "X_feats", phase)
        self.y_data_path = os.path.join(config.ASR_DATA_DIR_PATH, "Y_feats", phase)

    def __len__(self):
        return len(self.file_id_list)

    def __getitem__(self, idx):
        random.seed()

        # load input features
        x = np.load(os.path.join(self.x_data_path, self.file_id_list[idx]))
        x_len = len(x)

        start_idx = random.randint(0, x_len - config.INPUT_LENGTH)
        end_idx = start_idx + config.INPUT_LENGTH
        x = x[start_idx:end_idx]

        x = scale(x, self.X_mean, self.X_scale)
        x = x.T

        # load output features
        y = np.load(os.path.join(self.y_data_path, self.file_id_list[idx]))
        y = y[start_idx:end_idx]

        x, y = torch.from_numpy(x) , torch.from_numpy(y).long()

        return x, y

class SpeechGenDataset(Dataset):
    def __init__(self, phase, data, spk):
        # load scaler
        asr_scaler_file = os.path.join(config.ASR_DATA_DIR_PATH, "scaler.pkl")
        asr_scaler = pickle.load(open(asr_scaler_file, "rb"))
        if config.SCALE_PER_SPK:
            vc_scaler_file = os.path.join(config.VC_DATA_DIR_PATH, "scalers", spk, "scaler.pkl")
        else:
            vc_scaler_file = os.path.join(config.VC_DATA_DIR_PATH, "scaler.pkl")
        vc_scaler = pickle.load(open(vc_scaler_file, "rb"))
        self.X_mean, self.X_scale = asr_scaler["X_mean"], asr_scaler["X_scale"]
        self.Y_mean, self.Y_scale = vc_scaler["Y_mean"], vc_scaler["Y_scale"]

        # load files
        self.file_id_list = []
        file_id_list_path = os.path.join(config.VC_DATA_DIR_PATH, data + ".list")
        with open(file_id_list_path) as fp:
            for line in fp:
                line = line.strip()
                if spk in line:
                    self.file_id_list.append(line)

        self.x_data_path = os.path.join(config.VC_DATA_DIR_PATH, "X_feats", phase)
        self.y_data_path = os.path.join(config.VC_DATA_DIR_PATH, "Y_feats", phase)

    def __len__(self):
        return len(self.file_id_list)

    def __getitem__(self, idx):
        random.seed()

        # load input features
        x = np.load(os.path.join(self.x_data_path, self.file_id_list[idx]))
        x_len = len(x)

        start_idx = random.randint(0, x_len - config.INPUT_LENGTH)
        end_idx = start_idx + config.INPUT_LENGTH
        x = x[start_idx:end_idx]
        x = scale(x, self.X_mean, self.X_scale)
        x = x.T

        # load output features
        y = np.load(os.path.join(self.y_data_path, self.file_id_list[idx]))
        y = scale(y, self.Y_mean, self.Y_scale)
        y = y[start_idx:end_idx]
        y = y.T

        x, y = torch.from_numpy(x).float() , torch.from_numpy(y).float()

        return x, y

class SpeechGenMultiDataset(Dataset):
    def __init__(self, phase, data):
        # load scaler
        asr_scaler_file = os.path.join(config.ASR_DATA_DIR_PATH, "scaler.pkl")
        asr_scaler = pickle.load(open(asr_scaler_file, "rb"))
        vc_scaler_file = os.path.join(config.VC_DATA_DIR_PATH, "scaler.pkl")
        vc_scaler = pickle.load(open(vc_scaler_file, "rb"))
        self.X_mean, self.X_scale = asr_scaler["X_mean"], asr_scaler["X_scale"]
        self.Y_mean, self.Y_scale = vc_scaler["Y_mean"], vc_scaler["Y_scale"]

        # load files
        self.file_id_list = []
        file_id_list_path = os.path.join(config.VC_DATA_DIR_PATH, data + ".list")
        with open(file_id_list_path) as fp:
            for line in fp:
                line = line.strip()
                self.file_id_list.append(line)

        self.x_data_path = os.path.join(config.VC_DATA_DIR_PATH, "X_feats", phase)
        self.y_data_path = os.path.join(config.VC_DATA_DIR_PATH, "Y_feats", phase)

    def __len__(self):
        return len(self.file_id_list)

    def __getitem__(self, idx):
        random.seed()

        # load input features
        x = np.load(os.path.join(self.x_data_path, self.file_id_list[idx]))
        x_len = len(x)
        
        # get spaker index (e.g. jvs001/... -> 0)
        spk = int(self.file_id_list[idx].split("/")[0].replace("jvs", "")) - 1
        spk = F.one_hot(torch.tensor(spk), num_classes=config.N_SPEAKERS).float()

        try:
            start_idx = random.randint(0, x_len - config.INPUT_LENGTH)
        except Exception:
            print(self.file_id_list[idx])
            raise

        end_idx = start_idx + config.INPUT_LENGTH
        x = x[start_idx:end_idx]
        x = scale(x, self.X_mean, self.X_scale)
        x = x.T

        # load output features
        y = np.load(os.path.join(self.y_data_path, self.file_id_list[idx]))
        y = scale(y, self.Y_mean, self.Y_scale)
        y = y[start_idx:end_idx]
        y = y.T

        x, y = torch.from_numpy(x).float(), torch.from_numpy(y).float()

        return x, y, spk

if __name__ == "__main__":
    data = SpeechRecogDataset(phase = "train", data = "valid")
    loader = torch.utils.data.DataLoader(data, shuffle=True)
    for x, y in loader:
        print(x.shape)
        print(y.shape)
