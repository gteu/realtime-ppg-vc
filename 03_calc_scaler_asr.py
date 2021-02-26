# coding: utf-8
"""Calculate mean and variance of input features.

"""

import glob
import os
import pickle

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import config

def main():
    X_files = glob.glob(os.path.join(config.ASR_DATA_DIR_PATH, "X_feats", "train", "*", "*.npy"))
    standard_scaler = StandardScaler()

    for X_file in X_files:
        feature = np.load(X_file)
        standard_scaler.partial_fit(feature)
        print(X_file)

    X_mean = standard_scaler.mean_.astype(np.float32)
    X_var = standard_scaler.var_.astype(np.float32)
    X_scale = standard_scaler.scale_.astype(np.float32)

    scaler_file = os.path.join(config.ASR_DATA_DIR_PATH, "scaler.pkl")

    with open(scaler_file, "wb") as f:
        pickle.dump({"X_mean": X_mean, "X_var": X_var, "X_scale": X_scale}, f)

if __name__ == "__main__":
    main()