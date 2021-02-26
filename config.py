# coding: utf-8
import os

# config for preprating features
CSJ_DIR_PATH = "/common/db/CSJ"
JULIUS_BIN_PATH = "/opt/julius/bin/julius"
ASR_DATA_DIR_PATH = "./data/ASR"
VC_DATA_DIR_PATH = "./data/VC"
SR = 16000
NFFT = 512
WINDOW = 0.0125 # 16000 * 0.0125 = 200 samples
HOP = 0.005 # 16000 * 0.005 = 80 samples
NMELS = 40

# train/valid and test have different speakers.
#   ┌ TRAIN_P (ex. 0.9)
# ┌─┤
# │ └ VALID_P (ex. 0.1)
# │
# └── TEST_P  (ex. 0.1)

VALID_P = 0.1
TEST_P = 0.01

# config for training models
LOG_ROOT = "./runs"
GPU_ID = 0
BATCH_SIZE = 512
N_EPOCHS = 30
LR = 0.001
NUM_WORKERS = os.cpu_count()
SAVE_MODEL_FREQ = 10

# config for data and model
INPUT_LENGTH = 128
N_PHONE_CLASSES = 41

# config for test
INFERENCE_ROOT = "./runs/20210226_123930"