from datetime import datetime
import os
import glob
import random

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pyworld as pw
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pickle
from torch.nn import functional as F
import seaborn as sns
import pysptk
from nnmnkwii.postfilters import merlin_post_filter

import config
from model import SpeechRecogModel, SpeechGenModel
from utils.scale import scale
from scipy.io import wavfile

# target_f0 = 150 # JVS001
target_f0 = 220 # JSUT

def test(file_id, recog_model, gen_model, device, X_mean, X_scale, Y_mean, Y_scale):
    x_path = os.path.join(config.ASR_DATA_DIR_PATH, "X_feats", "train", file_id)

    # wav_file_path = glob.glob(os.path.join(config.ASR_DATA_DIR_PATH, "wav", file_id.replace(".npy", ".wav")))[0]
    wav_file_path = glob.glob(os.path.join(config.ASR_DATA_DIR_PATH, file_id.replace(".npy", ".wav")))[0]
    wav, _ = librosa.core.load(wav_file_path, config.SR_IN_OUT)

    wav = wav.astype(np.float64)

    _f0, t = pw.dio(wav, config.SR_IN_OUT)    # raw pitch extractor
    f0 = pw.stonemask(wav, _f0, t, config.SR_IN_OUT)  # pitch refinement
    ap = pw.d4c(wav, f0, t, config.SR_IN_OUT)
    # bap = pw.code_aperiodicity(ap, config.SR_IN_OUT)
    # ap = pw.decode_aperiodicity(bap.astype(np.float64), config.SR_IN_OUT, 2048)

    f0_rate = target_f0 / f0[np.nonzero(f0)].mean()

    # sp = pw.cheaptrick(wav, f0, t, config.SR)  # extract smoothed spectrogram
    # alpha =  pysptk.util.mcepalpha(config.SR)
    # mcep = pysptk.sp2mc(sp, config.N_MCEP, alpha)
    # sp = pysptk.mc2sp(mcep, fftlen = 1024, alpha = alpha)

    wav_mid, _ = librosa.core.load(wav_file_path, config.SR_MID)

    x = librosa.core.stft(
        y = wav_mid.astype(np.float32),
        n_fft = config.N_FFT,
        win_length = int(config.WINDOW * config.SR_MID),
        hop_length = int(config.HOP * config.SR_MID)
    )

    x = np.abs(x) ** 2
    mel_basis = librosa.filters.mel(sr = config.SR_MID, n_fft = config.N_FFT, n_mels = config.N_MELS)
    x = librosa.power_to_db(np.dot(mel_basis, x))
    x = x.T

    buf_nums = len(x) // config.INPUT_LENGTH

    y = np.array([], dtype=np.float64)

    for i in range(buf_nums):
        tmp_x = x[i * config.INPUT_LENGTH : (i+1) * config.INPUT_LENGTH]
        tmp_x = scale(tmp_x, X_mean, X_scale)
        tmp_x = torch.from_numpy(tmp_x).to(device)
        tmp_x = tmp_x.T
        tmp_x = tmp_x.unsqueeze(0)

        output = recog_model(tmp_x)
        output = F.softmax(output, dim=1)
        output = gen_model(output)

        # alpha =  pysptk.util.mcepalpha(config.SR_MID)
        alpha =  pysptk.util.mcepalpha(config.SR_IN_OUT)

        tmp_mcep = output.squeeze().cpu().detach().numpy().T.copy()
        tmp_mcep = Y_scale * tmp_mcep + Y_mean
        tmp_mcep = merlin_post_filter(tmp_mcep, alpha, fftlen = config.N_FFT * 2)

        tmp_f0 = f0[i * config.INPUT_LENGTH : (i+1) * config.INPUT_LENGTH] * f0_rate
        tmp_ap = ap[i * config.INPUT_LENGTH : (i+1) * config.INPUT_LENGTH]
        # tmp_sp = sp[i * config.INPUT_LENGTH : (i+1) * config.INPUT_LENGTH]
        tmp_sp = pysptk.mc2sp(tmp_mcep, fftlen = config.N_FFT * 2, alpha = alpha)

        tmp_y = pw.synthesize(tmp_f0.astype(np.float64), tmp_sp.astype(np.float64), tmp_ap.astype(np.float64), config.SR_IN_OUT)

        y = np.concatenate([y, tmp_y])

    # output_wav_path = os.path.join("gen_jsut", file_id.split("/")[-1].replace(".npy", ".wav"))
    output_wav_path = os.path.join("gen_jsut_16", file_id.split("/")[-1].replace(".npy", ".wav"))
    wavfile.write(output_wav_path, rate = config.SR_IN_OUT, data = y)

def main():
    device = "cuda:{}".format(config.GPU_ID) if torch.cuda.is_available() else "cpu"

    checkpoint_name = os.path.basename(config.ASR_INFERENCE_ROOT)
    
    recog_model_path = glob.glob(os.path.join(config.ASR_INFERENCE_ROOT, "*best.pth"))[0]
    gen_model_path = glob.glob(os.path.join(config.VC_INFERENCE_ROOT, "*best.pth"))[0]

    recog_model = SpeechRecogModel().to(device)
    gen_model = SpeechGenModel().to(device)
    recog_model.load_state_dict(torch.load(recog_model_path, map_location='cuda:0'))
    gen_model.load_state_dict(torch.load(gen_model_path, map_location='cuda:0'))

    recog_model.eval()
    gen_model.eval()

    asr_scaler_path = os.path.join(config.ASR_DATA_DIR_PATH, "scaler.pkl")
    vc_scaler_path = os.path.join(config.VC_DATA_DIR_PATH, "scaler.pkl")
    asr_scaler = pickle.load(open(asr_scaler_path, "rb"))
    vc_scaler = pickle.load(open(vc_scaler_path, "rb"))
    X_mean = asr_scaler["X_mean"]
    X_scale = asr_scaler["X_scale"]
    Y_mean = vc_scaler["Y_mean"]
    Y_scale = vc_scaler["Y_scale"]

    file_id_list = []
    # test_list_path = os.path.join(config.ASR_DATA_DIR_PATH, "valid.list")
    test_list_path = os.path.join(config.ASR_DATA_DIR_PATH, "file.list")
    with open(test_list_path) as f:
        for line in f:
            line = line.strip()
            file_id_list.append(line)

    # random.seed(0)
    # random.shuffle(file_id_list)
    for file_id in file_id_list[:50]:
        test(file_id, recog_model, gen_model, device, X_mean, X_scale, Y_mean, Y_scale)
        print(file_id)
        

if __name__ == "__main__":
    main()
