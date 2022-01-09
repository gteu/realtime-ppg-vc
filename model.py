# coding: utf-8

import torch
import torch.nn as nn
from torch.nn import functional as F

import config

class SpeechRecogModel(nn.Module):
    def __init__(self):
        super(SpeechRecogModel, self).__init__()
        # Feature extraction module
        self.conv_1 = nn.Conv1d(config.N_MELS, 256, 15, stride = 1, padding = 7)
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

class SpeechGenModel(nn.Module):
    def __init__(self):
        super(SpeechGenModel, self).__init__()
        self.conv_1 = nn.Conv1d(config.N_PHONE_CLASSES, 256, 15, stride = 1, padding = 7)
        self.bn_1 = nn.BatchNorm1d(256)
        self.conv_2 = nn.Conv1d(256, 512, 5, stride = 2, padding = 2)
        self.bn_2 = nn.BatchNorm1d(512)
        self.conv_3 = nn.Conv1d(512, 1024, 5, stride = 2, padding = 2)
        self.bn_3 = nn.BatchNorm1d(1024)
        self.deconv_1 = nn.ConvTranspose1d(1024, 512, 5, stride = 2, padding = 2, output_padding = 1)
        self.bn_4 = nn.BatchNorm1d(512)
        self.deconv_2 = nn.ConvTranspose1d(512, 256, 5, stride = 2, padding = 2, output_padding = 1)
        self.bn_5 = nn.BatchNorm1d(256)

        # use only static features 
        self.conv_4 = nn.Conv1d(256, config.N_MCEP, 15, stride = 1, padding = 7)
        # include dynamic features
        # self.conv_4 = nn.Conv1d(256, config.N_MCEP * 3, 15, stride = 1, padding = 7)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn_1(self.conv_1(x)))
        x = self.relu(self.bn_2(self.conv_2(x)))
        x = self.relu(self.bn_3(self.conv_3(x)))
        x = self.relu(self.bn_4(self.deconv_1(x)))
        x = self.relu(self.bn_5(self.deconv_2(x)))
        x = self.conv_4(x)

        return x

class SpeechGenMultiModel(nn.Module):
    def __init__(self):
        super(SpeechGenMultiModel, self).__init__()
        self.conv_1 = nn.Conv1d(config.N_PHONE_CLASSES, 256, 15, stride = 1, padding = 7)
        self.bn_1 = nn.BatchNorm1d(256)
        self.conv_2 = nn.Conv1d(256, 512, 5, stride = 2, padding = 2)
        self.bn_2 = nn.BatchNorm1d(512)
        self.conv_3 = nn.Conv1d(512, 1024, 5, stride = 2, padding = 2)
        self.bn_3 = nn.BatchNorm1d(1024)
        self.deconv_1 = nn.ConvTranspose1d(1024, 512, 5, stride = 2, padding = 2, output_padding = 1)
        self.bn_4 = nn.BatchNorm1d(512)
        self.deconv_2 = nn.ConvTranspose1d(512, 256, 5, stride = 2, padding = 2, output_padding = 1)
        self.bn_5 = nn.BatchNorm1d(256)

        # use only static features 
        self.conv_4 = nn.Conv1d(256, config.N_MCEP, 15, stride = 1, padding = 7)
        # include dynamic features
        # self.conv_4 = nn.Conv1d(256, config.N_MCEP * 3, 15, stride = 1, padding = 7)

        self.relu = nn.ReLU()

        self.spcode2input = nn.Sequential(
            nn.Linear(config.N_SPEAKERS, 128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, config.N_PHONE_CLASSES)
        )

    def forward(self, x, spcode):
        spcode = spcode.unsqueeze(1)
        spcode = spcode.expand(-1, x.shape[2], -1)
        spcode = self.spcode2input(spcode)
        spcode = spcode.transpose(1, 2)
        x = x + spcode

        x = self.relu(self.bn_1(self.conv_1(x)))
        x = self.relu(self.bn_2(self.conv_2(x)))
        x = self.relu(self.bn_3(self.conv_3(x)))
        x = self.relu(self.bn_4(self.deconv_1(x)))
        x = self.relu(self.bn_5(self.deconv_2(x)))
        x = self.conv_4(x)

        return x

class SpeechRecogGenModel(nn.Module):
    def __init__(self, recog_model, gen_model):
        super(SpeechRecogGenModel, self).__init__()
        self.recog_model = recog_model
        self.gen_model = gen_model

    def forward(self, x):
        x = self.recog_model(x)
        x = F.softmax(x, dim=1)
        x = self.gen_model(x)
        
        return x

class SpeechRecogGenMultiModel(nn.Module):
    def __init__(self, recog_model, gen_model):
        super(SpeechRecogGenMultiModel, self).__init__()
        self.recog_model = recog_model
        self.gen_model = gen_model

    def forward(self, x, spcode):
        x = self.recog_model(x)
        x = F.softmax(x, dim=1)
        x = self.gen_model(x, spcode)
        
        return x

class DiscrimintorModel(nn.Module):
    def __init__(self):
        super(DiscrimintorModel, self).__init__()
        self.conv_1 = nn.Conv1d(config.N_MCEP, 512, 1, stride = 1)
        self.bn_1 = nn.BatchNorm1d(512)
        self.conv_2 = nn.Conv1d(512, 512, 5, stride = 1, padding = 2)
        self.bn_2 = nn.BatchNorm1d(512)
        self.conv_3 = nn.Conv1d(512, 512, 5, stride = 1, padding = 2)
        self.bn_3 = nn.BatchNorm1d(512)
        self.conv_4 = nn.Conv1d(512, 1, 1, stride=1)
        # self.lin_1 = nn.Linear(128, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn_1(self.conv_1(x)))
        x = self.relu(self.bn_2(self.conv_2(x)))
        x = self.relu(self.bn_3(self.conv_3(x)))
        x = self.conv_4(x)

        # x = self.relu(self.conv_4(x))
        # x = self.lin_1(x)

        return x


if __name__ == "__main__":
    device = torch.device("cuda:{}".format(config.GPU_ID) if torch.cuda.is_available() else "cpu")
    # input shape: [batch size, feature dimension, feature length]
    sample = torch.randn(16, 40, 128).to(device)

    # test for single speaker model
    # sr_model = SpeechRecogModel().to(device)
    # sg_model = SpeechGenModel().to(device)
    # dis_model = DiscrimintorModel().to(device)
    # output = sr_model(sample)
    # output = F.softmax(output, dim=1)
    # output = sg_model(output)
    # output = dis_model(output)

    # test for multi speaker model
    sr_model = SpeechRecogModel().to(device)
    sg_model = SpeechGenMultiModel().to(device)
    dis_model = DiscrimintorModel().to(device)
    output = sr_model(sample)
    output = F.softmax(output, dim=1)
    spcode = torch.randn(16, 100).to(device)
    output = sg_model(output, spcode)

    print(output.shape)