# coding: utf-8
import glob
import os
import shutil

import librosa
from nnmnkwii.util import apply_delta_windows
import numpy as np
import pysptk
import pyworld as pw

import config
from utils.phone2num import phone2num

windows = [
    (0, 0, np.array([1.0])),
    (1, 1, np.array([-0.5, 0.0, 0.5])),
    (1, 1, np.array([1.0, -2.0, 1.0])),
]

def _extract_acoustic_feats_melspec(load_spk_dir, save_spk_dir):
    wav_file_paths = glob.glob(os.path.join(load_spk_dir, "*.wav"))
    for wav_file_path in wav_file_paths:
        file_id = os.path.splitext(os.path.basename(wav_file_path))[0]
        wav, _ = librosa.core.load(wav_file_path, config.SR)

        spec = librosa.core.stft(
            y = wav,
            n_fft = config.N_FFT,
            win_length = int(config.WINDOW * config.SR),
            hop_length = int(config.HOP * config.SR)
        )

        spec = np.abs(spec) ** 2
        mel_basis = librosa.filters.mel(sr = config.SR, n_fft = config.N_FFT, n_mels = config.N_MELS)
        spec = librosa.power_to_db(np.dot(mel_basis, spec))

        # [feats, time] -> [time, feats]
        spec = spec.T

        save_file_path = os.path.join(save_spk_dir, file_id + ".npy")
        np.save(save_file_path, spec)
        print(save_file_path)

def _extract_acoustic_feats_world(load_spk_dir, save_spk_dir):
    wav_file_paths = glob.glob(os.path.join(load_spk_dir, "*.wav"))
    for wav_file_path in wav_file_paths:
        file_id = os.path.splitext(os.path.basename(wav_file_path))[0]
        wav, _ = librosa.core.load(wav_file_path, config.SR)

        wav = wav.astype(np.float64)

        _f0, t = pw.dio(wav, config.SR)    # raw pitch extractor
        f0 = pw.stonemask(wav, _f0, t, config.SR)  # pitch refinement
        sp = pw.cheaptrick(wav, f0, t, config.SR)  # extract smoothed spectrogram

        alpha = pysptk.util.mcepalpha(config.SR)
        # static_mcep = pysptk.sp2mc(sp, config.N_MCEP, alpha)[:, 1:]
        static_mcep = pysptk.sp2mc(sp, config.N_MCEP - 1, alpha)
        static_delta_mcep = apply_delta_windows(static_mcep, windows) # [time, feats]

        save_file_path = os.path.join(save_spk_dir, file_id + ".npy")
        np.save(save_file_path, static_mcep)
        # np.save(save_file_path, static_delta_mcep)
        print(save_file_path)

def _extract_linguistic_feats(load_spk_dir, save_spk_dir):
    lab_file_paths = glob.glob(os.path.join(load_spk_dir, "*.lab"))
    for lab_file_path in lab_file_paths:
        file_id = os.path.splitext(os.path.basename(lab_file_path))[0]

        lab_feats = np.array([], dtype=int)
        with open(lab_file_path) as f:
            lines = f.readlines()
            for line in lines:
                splits = line.split()
                segment_start_time = int(splits[0][:4].replace(".", ""))
                segment_end_time = int(splits[1][:4].replace(".", ""))
                num_segment_frame = (segment_end_time - segment_start_time) * 2
                phone_num = phone2num[splits[2]]
                lab_feat = np.array([phone_num] * num_segment_frame)
                lab_feats = np.concatenate([lab_feats, lab_feat])

        save_file_path = os.path.join(save_spk_dir, file_id + ".npy")
        np.save(save_file_path, lab_feats)
        print(save_file_path)

def _normalize_feature_length(x_feats_spk_dir, y_feats_spk_dir):
    x_feats_file_paths = glob.glob(os.path.join(x_feats_spk_dir, "*.npy"))
    for x_feats_file_path in x_feats_file_paths:
        file_id = os.path.splitext(os.path.basename(x_feats_file_path))[0]
        y_feats_file_path = os.path.join(y_feats_spk_dir, file_id + ".npy")

        x_feats = np.load(x_feats_file_path)
        y_feats = np.load(y_feats_file_path)

        if x_feats.shape[0] == 0 or y_feats.shape[0] == 0:
            os.remove(x_feats_file_path)
            os.remove(y_feats_file_path)
            print("remove {} and {}".format(x_feats_file_path, y_feats_file_path))

        else:
            if x_feats.shape[0] > y_feats.shape[0]:
                x_feats = x_feats[:y_feats.shape[0]]
                np.save(x_feats_file_path, x_feats)
                print(x_feats_file_path)
            elif x_feats.shape[0] < y_feats.shape[0]:
                y_feats = y_feats[:x_feats.shape[0]]
                np.save(y_feats_file_path, y_feats)
                print(y_feats_file_path)
            else:
                print("Neither files are changed. - {}".format(file_id))

def split_data(data_dir_path):
    """Split data into train/valid data and test data.
    
    """

    # split data into train/valid data and test data
    spks = sorted(os.listdir(os.path.join(data_dir_path, "X_feats")))
    n_train_spks = int(len(spks) * (1 - config.TEST_P))
    train_spks = spks[:n_train_spks]
    test_spks = spks[n_train_spks:]

    for feat_name in ["X_feats", "Y_feats"]:
        train_dir = os.path.join(data_dir_path, feat_name, "train")
        test_dir = os.path.join(data_dir_path, feat_name, "test")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        for spk in train_spks:
            if spk in ["train", "test"]:
                continue
            from_spk_dir = os.path.join(data_dir_path, feat_name, spk)
            to_spk_dir = os.path.join(data_dir_path, feat_name, "train", spk)
            shutil.move(from_spk_dir, to_spk_dir)

        for spk in test_spks:
            if spk in ["train", "test"]:
                continue
            from_spk_dir = os.path.join(data_dir_path, feat_name, spk)
            to_spk_dir = os.path.join(data_dir_path, feat_name, "test", spk)
            shutil.move(from_spk_dir, to_spk_dir)

    # generate file id list 
    train_list_path = os.path.join(data_dir_path, "train.list")
    valid_list_path = os.path.join(data_dir_path, "valid.list")
    test_list_path = os.path.join(data_dir_path, "test.list")

    train_list, valid_list, test_list = [], [], []

    for i, spk in enumerate(train_spks):
        files = sorted(glob.glob(os.path.join(data_dir_path, "X_feats", "train", spk, "*.npy")))
        for j, file_path in enumerate(files):
            file_id = os.path.join(file_path.split("/")[-2], file_path.split("/")[-1])
            if j < len(files) * (1 - config.VALID_P):
                train_list.append(file_id)
            else:
                valid_list.append(file_id)

    for i, spk in enumerate(test_spks):
        files = sorted(glob.glob(os.path.join(data_dir_path, "X_feats", "test", spk, "*.npy")))
        for j, file_path in enumerate(files):
            file_id = os.path.join(file_path.split("/")[-2], file_path.split("/")[-1])
            test_list.append(file_id)

    with open(train_list_path, mode = "w") as f:
        f.write("\n".join(train_list))
    with open(valid_list_path, mode = "w") as f:
        f.write("\n".join(valid_list))
    with open(test_list_path, mode = "w") as f:
        f.write("\n".join(test_list))
