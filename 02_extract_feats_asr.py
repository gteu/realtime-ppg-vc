# coding: utf-8
"""Extract features for a speech recognition model.
extract_acoustic_feats() -> Extract acoustic features (input).
extract_linguistic_feats() -> Extract linguistic features (output)

"""

import glob
from multiprocessing import Pool
import os
import shutil

import numpy as np

import config
from utils.arg_wrapper import argwrapper
from utils.prepare_features import _extract_acoustic_feats_melspec, _extract_linguistic_feats, _normalize_feature_length, split_data

def extract_acoustic_feats():
    """Extract acoustic features (input).
    
    """
    spks = os.listdir(os.path.join(config.ASR_DATA_DIR_PATH, "wav"))

    func_args = []
    for spk in spks:
        load_spk_dir = os.path.join(config.ASR_DATA_DIR_PATH, "wav", spk)
        save_spk_dir = os.path.join(config.ASR_DATA_DIR_PATH, "X_feats", spk)
        os.makedirs(save_spk_dir, exist_ok=True)
        func_args.append([_extract_acoustic_feats_melspec, load_spk_dir, save_spk_dir])

    with Pool(os.cpu_count()) as p:
        for i, _ in enumerate(p.imap_unordered(argwrapper, func_args)):
            pass

def extract_linguistic_feats():
    """Extract linguistic features (output).
    
    """
    spks = os.listdir(os.path.join(config.ASR_DATA_DIR_PATH, "wav"))

    func_args = []
    for spk in spks:
        load_spk_dir = os.path.join(config.ASR_DATA_DIR_PATH, "lab", spk)
        save_spk_dir = os.path.join(config.ASR_DATA_DIR_PATH, "Y_feats", spk)
        os.makedirs(save_spk_dir, exist_ok=True)
        func_args.append([_extract_linguistic_feats, load_spk_dir, save_spk_dir])

    with Pool(os.cpu_count()) as p:
        for i, _ in enumerate(p.imap_unordered(argwrapper, func_args)):
            pass

def normalize_feature_length():
    """Make the length of acoustic feats and linguistic feats the same.
    
    """
    
    spks = os.listdir(os.path.join(config.ASR_DATA_DIR_PATH, "wav"))

    func_args = []
    for spk in spks:
        x_feats_spk_dir = os.path.join(config.ASR_DATA_DIR_PATH, "X_feats", spk)
        y_feats_spk_dir = os.path.join(config.ASR_DATA_DIR_PATH, "Y_feats", spk)
        func_args.append([_normalize_feature_length, x_feats_spk_dir, y_feats_spk_dir])

    with Pool(os.cpu_count()) as p:
        for i, _ in enumerate(p.imap_unordered(argwrapper, func_args)):
            pass

if __name__ == "__main__":
    extract_acoustic_feats()
    extract_linguistic_feats()
    normalize_feature_length()
    split_data(data_dir_path = config.ASR_DATA_DIR_PATH)