# BERT-Article-Recommender-with-Faiss
* BERTとfaissを用いた記事推薦の実装

# 実験方法
```
$ conda create -n allennlp python=3.7
$ conda activate allennlp
$ pip install -r requirements.txt
$ download.sh

# debug用 3ラベルのみ
$ python3 main.py -debug True

# 本番実験
$ python3 main.py
```