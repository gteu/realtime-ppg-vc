# RealtimePPGVC

Voice conversion model for real-time synthesis using PPG (Phonetic PosteriorGram) as an intermediate feature, written in Pytorch.

## Implementation details
* Reference
  * [Joint Adversarial Training of Speech Recognition and Synthesis
Models for Many-to-One Voice Conversion Using Phonetic
Posteriorgrams](https://www.jstage.jst.go.jp/article/transinf/E103.D/9/E103.D_2019EDP7297/_pdf/-char/en) 
<!--   *  [Implementation of DNN-based real-time voice conversion and its
improvements by audio data augmentation and mask-shaped device](https://www.isca-speech.org/archive/pdfs/ssw_2019/arakawa19_ssw.pdf)  -->
<!-- * 参考リンク（参考文献2のリアルタイム実装解説記事）  
https://engineer.dena.com/posts/2020.03/voice-conversion-for-entertainment/
* 参考リンク（リアルタイムVC解説記事）  
https://blog.hiroshiba.jp/realtime-yukarin-introduction/ -->
* Dataset
  * [CSJ Corpus (for training PPG)](https://ccd.ninjal.ac.jp/csj/)  
  * [JVS Corpus (for target speech)](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus)  
<!-- * モデル
  * マルチフレーム入力＋feed-forwardで音声認識が動けばそれを用いる（恐らく一番Latencyが少ない）  
  * feed-forwardでなく1D-CNNを用いる
  * 余裕があれば、https://arxiv.org/pdf/1811.06621.pdf にあるStreaming RNN-Tを用いる -->

## PPG training result
<img src="https://user-images.githubusercontent.com/25415810/109280313-34060200-785e-11eb-8279-8eef5c738330.png" width="400px">

## PPG training sample
* Transcript(JP)
  * 「二回から...」
* Transcript(EN)
  * "n i k a i k a r a ..."  
<img src="https://user-images.githubusercontent.com/25415810/109280300-2ea8b780-785e-11eb-9085-776201644f36.png" width="1200px">

The correspondence between index and phone is described [here](/utils/phone2num.py).

<!-- ## VC音声サンプル(ベースライン)
https://drive.google.com/drive/folders/1EYnLiFi--erwxPRzNRpuI1AAjfAScyxs?usp=sharing -->
