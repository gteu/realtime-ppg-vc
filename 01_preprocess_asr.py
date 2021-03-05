# coding: utf-8
"""Preprocess dataset.
split_wav() -> Split wav files into short segments that Julius can process.
align_wav_and_text() -> Align wav and text using julius.

"""

import glob
from multiprocessing import Pool
import os

import config
from utils.preprocess import append_segment, remove_unnecessary_tag, load_trn, yomi2phone
from utils.arg_wrapper import argwrapper

def save_wav_and_text(segments, in_wav_file_path, file_id):
    """Save splitted wav and text.
    
    """

    for i, segment in enumerate(segments):
        # save wav
        os.makedirs(os.path.join(config.ASR_DATA_DIR_PATH, "wav", file_id), exist_ok=True)
        out_wav_file_path = os.path.join(config.ASR_DATA_DIR_PATH, "wav", file_id, file_id + "_" + str(i).zfill(3) + ".wav")
        os.system("sox {} {} trim {} {}".format(in_wav_file_path,\
                                                out_wav_file_path,\
                                                float(segment["segment_start_time"]),\
                                                float(segment["segment_end_time"]) - float(segment["segment_start_time"])))
        print("Output {}".format(out_wav_file_path))

        # save text
        os.makedirs(os.path.join(config.ASR_DATA_DIR_PATH, "txt", file_id), exist_ok=True)
        out_text_file_path = os.path.join(config.ASR_DATA_DIR_PATH, "txt", file_id, file_id + "_" + str(i).zfill(3) + ".txt")
        with open(out_text_file_path, "w") as f:
            f.write(segment["segment_text"])

        print("Output {}".format(out_text_file_path))

def _split_wav(trn_file_path, wav_file_path, file_id):
    try:
        segments = load_trn(trn_file_path)
        save_wav_and_text(segments, wav_file_path, file_id)
    except:
        pass

def split_wav():
    """Split wav in silent sections.    
    
    """
    wav_file_paths = glob.glob(os.path.join(config.CSJ_DIR_PATH, "*/*/*.wav"))
    func_args = []
    for wav_file_path in wav_file_paths:
        trn_file_path = wav_file_path.replace("-L", "").replace("-R", "").replace(".wav", ".trn")
        file_id = os.path.splitext(os.path.basename(wav_file_path))[0].replace("-L", "").replace("-R", "")
        func_args.append([_split_wav, trn_file_path, wav_file_path, file_id])
    
    with Pool(os.cpu_count()) as p:
        for i, _ in enumerate(p.imap_unordered(argwrapper, func_args)):
            pass

def _align_wav_and_text(wav_spk_dir_path, txt_spk_dir_path, lab_spk_dir_path):
    os.system("perl ./utils/segmentation-kit/dir_change_segment_julius.pl {} {} {}".format(wav_spk_dir_path, txt_spk_dir_path, lab_spk_dir_path))

def align_wav_and_text():
    """Align wav and text.

    """
    wav_spk_dir_paths = glob.glob(os.path.join(config.ASR_DATA_DIR_PATH, "wav", "*"))
    func_args = []
    for wav_spk_dir_path in wav_spk_dir_paths:
        txt_spk_dir_path = os.path.join(config.ASR_DATA_DIR_PATH, "txt", os.path.basename(wav_spk_dir_path))
        lab_spk_dir_path = os.path.join(config.ASR_DATA_DIR_PATH, "lab", os.path.basename(wav_spk_dir_path))
        os.makedirs(lab_spk_dir_path, exist_ok=True)
        func_args.append([_align_wav_and_text, wav_spk_dir_path, txt_spk_dir_path, lab_spk_dir_path])

    with Pool(os.cpu_count()) as p:
        for i, _ in enumerate(p.imap_unordered(argwrapper, func_args)):
            pass

def prepare_csj():
    split_wav()
    align_wav_and_text()
    
if __name__ == "__main__":
    prepare_csj()
