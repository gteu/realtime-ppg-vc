# coding: utf-8
"""Extract features for voice conversion model.
extract_input_acoustic_feats() -> Extract input acoustic features.
extract_output_acoustic_feats() -> Extract output acoustic features.

"""

import glob
from multiprocessing import Pool
import os

import config

from utils.arg_wrapper import argwrapper
from utils.prepare_features import _extract_acoustic_feats_melspec, _extract_acoustic_feats_world, _normalize_feature_length, split_data

def extract_input_acoustic_feats():
    """Extract input acoustic features (Mel-spectrogram).
    
    """
    spks = [os.path.basename(spk_path) for spk_path in glob.glob(os.path.join(config.JVS_DIR_PATH, "jvs*"))]
    spks = sorted(spks)

    func_args = []
    for spk in spks:
        load_spk_dir = os.path.join(config.JVS_DIR_PATH, spk, "parallel100", "wav24kHz16bit")
        save_spk_dir = os.path.join(config.VC_DATA_DIR_PATH, "X_feats", spk)
        os.makedirs(save_spk_dir, exist_ok=True)
        func_args.append([_extract_acoustic_feats_melspec, load_spk_dir, save_spk_dir])

    with Pool(os.cpu_count()) as p:
        for i, _ in enumerate(p.imap_unordered(argwrapper, func_args)):
            pass

def extract_output_acoustic_feats():
    """Extract output acoustic features (WORLD features).
    
    """
    spks = [os.path.basename(spk_path) for spk_path in glob.glob(os.path.join(config.JVS_DIR_PATH, "jvs*"))]
    spks = sorted(spks)

    func_args = []
    for spk in spks:
        load_spk_dir = os.path.join(config.JVS_DIR_PATH, spk, "parallel100", "wav24kHz16bit")
        save_spk_dir = os.path.join(config.VC_DATA_DIR_PATH, "Y_feats", spk)
        os.makedirs(save_spk_dir, exist_ok=True)
        func_args.append([_extract_acoustic_feats_world, load_spk_dir, save_spk_dir])

    with Pool(os.cpu_count()) as p:
        for i, _ in enumerate(p.imap_unordered(argwrapper, func_args)):
            pass

def normalize_feature_length():
    """Make the length of acoustic feats and linguistic feats the same.
    
    """
    
    spks = [os.path.basename(spk_path) for spk_path in glob.glob(os.path.join(config.JVS_DIR_PATH, "jvs*"))]

    func_args = []
    for spk in spks:
        x_feats_spk_dir = os.path.join(config.VC_DATA_DIR_PATH, "X_feats", spk)
        y_feats_spk_dir = os.path.join(config.VC_DATA_DIR_PATH, "Y_feats", spk)
        func_args.append([_normalize_feature_length, x_feats_spk_dir, y_feats_spk_dir])

    with Pool(os.cpu_count()) as p:
        for i, _ in enumerate(p.imap_unordered(argwrapper, func_args)):
            pass

if __name__ == "__main__":
    extract_input_acoustic_feats()
    extract_output_acoustic_feats()
    normalize_feature_length()
    split_data(data_dir_path = config.VC_DATA_DIR_PATH)