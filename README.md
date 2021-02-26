# RealtimePPGVC

## 実装詳細（予定）
* 参考論文1（Realtime実装）  
https://www.isca-speech.org/archive/SSW_2019/pdfs/SSW10_P_1-10.pdf  
* 参考論文2（PPGVC実装）  
http://sython.org/papers/ASJ/saito2019asja_dena.pdf
* 参考リンク（参考文献2のリアルタイム実装解説記事）  
https://engineer.dena.com/posts/2020.03/voice-conversion-for-entertainment/
* 参考リンク（リアルタイムVC解説記事）  
https://blog.hiroshiba.jp/realtime-yukarin-introduction/
* 使用データセット  
  * 音声認識：[CSJコーパス](https://pj.ninjal.ac.jp/corpus_center/csj/)  
  * 声質変換のターゲット：[JVSコーパス](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus)  
* モデル
  * マルチフレーム入力＋feed-forwardで音声認識が動けばそれを用いる（恐らく一番Latencyが少ない）  
  * feed-forwardでなく1D-CNNを用いる
  * 余裕があれば、https://arxiv.org/pdf/1811.06621.pdf にあるStreaming RNN-Tを用いる
* To-do
  * 音声認識部分が正しく動くことを確認する

## PPG認識学習結果
<img src="https://user-images.githubusercontent.com/25415810/109280313-34060200-785e-11eb-8279-8eef5c738330.png" width="400px">

## PPG学習サンプル
* テキスト 「二回から三回...」
<img src="https://user-images.githubusercontent.com/25415810/109280300-2ea8b780-785e-11eb-9085-776201644f36.png" width="1200px">
