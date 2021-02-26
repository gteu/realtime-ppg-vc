# coding: utf-8

import torch
import torch.nn as nn
from torch.nn import functional as F

import config

class SpeechRecogModel(nn.Module):
    def __init__(self):
        super(SpeechRecogModel, self).__init__()
        # Feature extraction module
        self.conv_1 = nn.Conv1d(config.NMELS, 256, 15, stride = 1, padding = 7)
        self.bn_1 = nn.BatchNorm1d(256)
        self.conv_2 = nn.Conv1d(256, 512, 5, stride = 2, padding = 2)
        self.bn_2 = nn.BatchNorm1d(512)
        self.conv_3 = nn.Conv1d(512, 1024, 5, stride = 2, padding = 2)
        self.bn_3 = nn.BatchNorm1d(1024)
        self.deconv_1 = nn.ConvTranspose1d(1024, 512, 5, stride = 2, padding = 2, output_padding = 1)
        self.bn_4 = nn.BatchNorm1d(512)
        self.deconv_2 = nn.ConvTranspose1d(512, 256, 5, stride = 2, padding = 2, output_padding = 1)
        self.bn_5 = nn.BatchNorm1d(256)

        # Phoneme prediction module
        self.conv_4 = nn.Conv1d(256, config.N_PHONE_CLASSES, 15, stride = 1, padding = 7)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn_1(self.conv_1(x)))
        x = self.relu(self.bn_2(self.conv_2(x)))
        x = self.relu(self.bn_3(self.conv_3(x)))
        x = self.relu(self.bn_4(self.deconv_1(x)))
        x = self.relu(self.bn_5(self.deconv_2(x)))
        x = self.conv_4(x)

        return x
    

if __name__ == "__main__":
    device = torch.device("cuda:{}".format(config.GPU_ID) if torch.cuda.is_available() else "cpu")
    # input shape: [batch size, feature dimension, feature length]
    sample = torch.randn(16, 40, 128).to(device)
    sr_model = SpeechRecogModel().to(device)
    print(sr_model(sample).shape)