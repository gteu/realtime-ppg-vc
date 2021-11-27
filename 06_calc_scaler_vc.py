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
    if config.SCALE_PER_SPK:
        spks = [os.path.basename(spk_path) for spk_path in glob.glob(os.path.join(config.VC_DATA_DIR_PATH, "Y_feats", "train", "jvs*"))]
        for spk in spks:
            Y_files = glob.glob(os.path.join(config.VC_DATA_DIR_PATH, "Y_feats", "train", spk, "*.npy"))
            standard_scaler = StandardScaler()

            for Y_file in Y_files:
                feature = np.load(Y_file)
                standard_scaler.partial_fit(feature)
                print(Y_file)

            Y_mean = standard_scaler.mean_.astype(np.float32)
            Y_var = standard_scaler.var_.astype(np.float32)
            Y_scale = standard_scaler.scale_.astype(np.float32)

            os.makedirs(os.path.join(config.VC_DATA_DIR_PATH, "scalers", spk), exist_ok=True)
            scaler_file = os.path.join(config.VC_DATA_DIR_PATH, "scalers", spk, "scaler.pkl")

            with open(scaler_file, "wb") as f:
                pickle.dump({"Y_mean": Y_mean, "Y_var": Y_var, "Y_scale": Y_scale}, f)

    else:
        Y_files = glob.glob(os.path.join(config.VC_DATA_DIR_PATH, "Y_feats", "train", "*", "*.npy"))
        standard_scaler = StandardScaler()

        for Y_file in Y_files:
            feature = np.load(Y_file)
            standard_scaler.partial_fit(feature)
            print(Y_file)

        Y_mean = standard_scaler.mean_.astype(np.float32)
        Y_var = standard_scaler.var_.astype(np.float32)
        Y_scale = standard_scaler.scale_.astype(np.float32)

        scaler_file = os.path.join(config.VC_DATA_DIR_PATH, "scaler.pkl")

        with open(scaler_file, "wb") as f:
            pickle.dump({"Y_mean": Y_mean, "Y_var": Y_var, "Y_scale": Y_scale}, f)
        

if __name__ == "__main__":
    main()
