# RealtimePPGVC

Voice conversion model for real-time synthesis using PPG (Phonetic PosteriorGram) as an intermediate feature, written in Pytorch. 

## Implementation details
### Reference
  * [Joint Adversarial Training of Speech Recognition and Synthesis
Models for Many-to-One Voice Conversion Using Phonetic
Posteriorgrams](https://www.jstage.jst.go.jp/article/transinf/E103.D/9/E103.D_2019EDP7297/_pdf/-char/en) 

<!--   *  [Implementation of DNN-based real-time voice conversion and its
improvements by audio data augmentation and mask-shaped device](https://www.isca-speech.org/archive/pdfs/ssw_2019/arakawa19_ssw.pdf)  -->
<!-- * 参考リンク（参考文献2のリアルタイム実装解説記事）  
https://engineer.dena.com/posts/2020.03/voice-conversion-for-entertainment/
* 参考リンク（リアルタイムVC解説記事）  
https://blog.hiroshiba.jp/realtime-yukarin-introduction/ -->
### Dataset
  * [CSJ Corpus (for training PPG)](https://ccd.ninjal.ac.jp/csj/)  
  * [JVS Corpus (for target speech)](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus)  

## PPG training result
<img src="https://user-images.githubusercontent.com/25415810/109280313-34060200-785e-11eb-8279-8eef5c738330.png" width="400px">

## PPG training sample
### Transcript (Japanese)
  * 「二回から...」
### Transcript (English)
  * "n i k a i k a r a ..."  
<img src="https://user-images.githubusercontent.com/25415810/109280300-2ea8b780-785e-11eb-9085-776201644f36.png" width="1200px">

The correspondence between index and phone is described [here](/utils/phone2num.py).

## VC speech sample
Baseline samples (No GAN, No DAT)

https://drive.google.com/drive/folders/1Djq4dwZgJdGy4rFVArZY_kLySoxu9iSj?usp=sharing 

- `gen_[ID].wav`: generated speech
- `ref_[ID].wav`: source speech
- `jsut_target.wav`: speech from target speaker

