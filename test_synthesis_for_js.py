import json
import librosa
import numpy as np
import onnx
import onnxruntime as ort
import os
import pickle
import pysptk

import config
from utils.scale import scale

asr_scaler_path = os.path.join(config.ASR_DATA_DIR_PATH, "scaler.pkl")
vc_scaler_path = os.path.join(config.VC_DATA_DIR_PATH, "scaler.pkl")
asr_scaler = pickle.load(open(asr_scaler_path, "rb"))
vc_scaler = pickle.load(open(vc_scaler_path, "rb"))
X_mean = asr_scaler["X_mean"]
X_scale = asr_scaler["X_scale"]
Y_mean = vc_scaler["Y_mean"]
Y_scale = vc_scaler["Y_scale"]

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
            np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)): #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def write_json(path, object):
    with open(path, 'w') as f:
        f.write(json.dumps(object, cls=NumpyEncoder))
        print("Saved {}.\nShape: {}".format(path, np.array(object).shape))
  
def array_splitter(arr, frame_len):
    arrs = []
    n_frames = len(arr) // frame_len
    for i in range(n_frames):
        arrs.append(arr[i * frame_len: (i+1) * frame_len])

    return np.array(arrs)

if __name__ == "__main__":
    # load wav
    wav_path = "ref_test_synthesis_for_js.wav"
    wav_all, _ = librosa.core.load(wav_path, config.SR)
    wavs = array_splitter(wav_all, frame_len = int(config.SR * 0.64))
    write_json("jsons/source_wav.json", wavs)

    # wav to mel
    mels = []
    for wav in wavs:
        wav = wav[:-1]
        mel = librosa.feature.melspectrogram(wav, center=True, sr=16000, n_fft=512, hop_length=80, n_mels=40)
        mels.append(mel)
    write_json("jsons/source_melspec.json", mels)
    
    # mel to mel (db)
    mel_dbs = []
    for mel in mels:
        mel_db = librosa.power_to_db(mel)
        mel_dbs.append(mel_db)
    write_json("jsons/source_melspec_db.json", mel_dbs)
    
    # mel (db) to scaled mel (db)
    mel_db_scaleds = []
    for mel_db in mel_dbs:
        mel_db_scaled = scale(mel_db.T, X_mean, X_scale).T
        mel_db_scaleds.append(mel_db_scaled)
    write_json("jsons/source_melspec_db_scaled.json", mel_db_scaleds)

    # start onnx session
    session_options = ort.SessionOptions()
    session_options.log_severity_level = 3
    session = ort.InferenceSession('ppgvc.onnx', session_options)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    outputs = []
    for i, mel_db_scaled in enumerate(mel_db_scaleds):
        output = session.run([output_name], {input_name: mel_db_scaleds[i][np.newaxis, :, :]})[0].squeeze()
        outputs.append(output)
    write_json("jsons/target_mcep_scaled.json", outputs)

    # scaled mcep to mcep
    mceps = []
    for output in outputs:
        mcep = (Y_scale * output.T + Y_mean).T
        mceps.append(mcep)
    write_json("jsons/target_mcep.json", mceps)

    # mcep to sp
    sps = []
    for mcep in mceps:
        sp = pysptk.mc2sp(mcep, fftlen = 1024, alpha = pysptk.util.mcepalpha(config.SR))
        sps.append(sp)
    write_json("jsons/target_sp.json", sps)