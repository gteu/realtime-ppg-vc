# coding: utf-8

import codecs
import re

import jaconv

def append_segment(segments_list, segment_start_time, segment_end_time, segment_text):
    segments_list.append({"segment_start_time": segment_start_time,\
                          "segment_end_time": segment_end_time,\
                          "segment_text": segment_text})

    return segments_list

def remove_unnecessary_tag(kata):
    if "×" in kata:
        return None
    ret_kata = kata
    # remove pronunciation error tag (W)
    for kata_w in re.findall("\(W.*\)", kata):
        kata_w_removed = kata_w.replace("(W", "").replace(")", "")    
        kata_w_removed = kata_w_removed.split(";")[0]
        ret_kata = ret_kata.replace(kata_w, kata_w_removed)
    # remove filler tag (F)
    for kata_f in re.findall("\(F.*\)", ret_kata):
        kata_f_removed = kata_f.replace("(F", "").replace(")", "")
        ret_kata = ret_kata.replace(kata_f, kata_f_removed)
    # remove double pronunciation tag (D)
    for kata_d in re.findall("\(D.*\)", ret_kata):
        kata_d_removed = kata_d.replace("(D", "").replace(")", "")
        ret_kata = ret_kata.replace(kata_d, kata_d_removed)
    # remove <> tag
    for tag in re.findall("<.*>", ret_kata):
        ret_kata = ret_kata.replace(tag, "")

    # remove the other tags that appear infrequently
    if "(" in ret_kata:
        ret_kata = None

    return ret_kata

def load_trn(trn_path):
    """Load transctipt file (*.trn).

    """
    
    segments = []

    with codecs.open(trn_path, "r", "shift_jis") as f:
        lines = f.readlines()

    segment_num = 1
    segment_start_time, segment_end_time = None, None
    segment_to_be_used = True
    
    # load the file line by line, and split segments
    for line in lines:
        if line.split()[0] == str(segment_num).zfill(4):
            if segment_to_be_used and segment_start_time != None and segment_end_time != None and segment_text != "":
                segments = append_segment(segments, segment_start_time, segment_end_time, segment_text)
            segment_start_time = line.split()[1].split("-")[0]
            segment_end_time = line.split()[1].split("-")[1]
            segment_text = ""
            segment_to_be_used = True
            segment_num += 1
        elif len(line.split("&")) > 1: 
            kata = line.split("&")[1].replace(" ", "").rstrip()
            kata = remove_unnecessary_tag(kata)
            if kata == None:
                segment_to_be_used = False    
            else:
                hira = jaconv.kata2hira(kata)
                segment_text += hira

    segments = append_segment(segments, segment_start_time, segment_end_time, segment_text)

    segments_concat = []

    # concatenate consecutive segments
    segment_start_time = segments[0]["segment_start_time"]
    segment_end_time = segments[0]["segment_end_time"]
    segment_text = segments[0]["segment_text"]
    for i in range(len(segments) - 1):
        silence_time = float(segments[i+1]["segment_start_time"]) - float(segments[i]["segment_end_time"])
        if silence_time > 0.5:
            if len(segment_text) != 0 and 15 > float(segment_end_time) - float(segment_start_time) > 1:
                if segment_text[-4:] == " sp ":
                    segment_text = segment_text[:-4]
                segments_concat = append_segment(segments_concat, segment_start_time, segment_end_time, segment_text)
            segment_text = segments[i+1]["segment_text"]
            segment_start_time = segments[i+1]["segment_start_time"]
            segment_end_time = segments[i+1]["segment_end_time"]
        else:
            segment_text += segments[i+1]["segment_text"] + " sp "
            segment_end_time = segments[i+1]["segment_end_time"]

    if len(segment_text) != 0 and 15 > float(segment_end_time) - float(segment_start_time) > 1:
        segments_concat = append_segment(segments_concat, segment_start_time, segment_end_time, segment_text)
 
    return segments_concat

def yomi2phone(yomi):
    from yomi2phone_rules import three_chars_yomi2phone, two_chars_yomi2phone, one_char_yomi2phone
    # phone = "silB"
    phone = ""
    while len(yomi):
        if len(yomi) > 2:
            tmp_yomi = yomi[:3]
            if tmp_yomi in three_chars_yomi2phone.keys():
                phone += three_chars_yomi2phone[tmp_yomi]
                yomi = yomi[3:]
                continue
        
        if len(yomi) > 1:
            tmp_yomi = yomi[:2]
            if tmp_yomi in two_chars_yomi2phone.keys():
                phone += two_chars_yomi2phone[tmp_yomi]
                yomi = yomi[2:]
                continue

        if len(yomi) > 0:
            tmp_yomi = yomi[0]
            if tmp_yomi in one_char_yomi2phone.keys():
                phone += one_char_yomi2phone[tmp_yomi]
                yomi = yomi[1:]
                continue
        
    # phone += " silE"

    return phone
