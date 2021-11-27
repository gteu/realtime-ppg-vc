# coding: utf-8
import os

# config for preprating features
CSJ_DIR_PATH = "/common/db/CSJ"
JVS_DIR_PATH = "/common/db/JVS/jvs_ver1"
JULIUS_BIN_PATH = "/opt/julius/bin/julius"
ASR_DATA_DIR_PATH = "./data/ASR"
VC_DATA_DIR_PATH = "./data/VC"
SR = 16000
N_FFT = 512
WINDOW = 0.0125 # 16000 * 0.0125 = 200 samples
HOP = 0.005 # 16000 * 0.005 = 80 samples
N_MELS = 40
N_MCEP = 40
N_PHONE_CLASSES = 41
INPUT_LENGTH = 128
SCALE_PER_SPK = True 

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
NUM_WORKERS = os.cpu_count()
SAVE_MODEL_FREQ = 10

ASR_BATCH_SIZE = 512
ASR_N_EPOCHS = 10
ASR_LR = 0.001

VC_SPEAKER = "jvs001"
VC_BATCH_SIZE = 4
VC_N_EPOCHS = 1000
VC_LR = 0.0001

# config for test
ASR_INFERENCE_ROOT = "./runs/ASR/20210325_024244"
VC_INFERENCE_ROOT = "./runs/VC/20210325_031200"
