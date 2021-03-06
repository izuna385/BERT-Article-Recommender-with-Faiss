# BERT-Article-Recommender-with-Faiss
* BERTとfaissを用いた記事推薦の実装

# モデル
![dummy](./img/model.png)

# 環境設定
```
$ conda create -n allennlp python=3.7
$ conda activate allennlp
$ pip install -r requirements.txt
$ download.sh
```

# 実験方法
$ `python3 main.py -num_epochs 10`

* 特にパラメータ調整は行っていない。dev acc. ~85%, test acc. ~83%.

# 記事
[https://qiita.com/izuna385/items/a004982f76256cab1b2d](https://qiita.com/izuna385/items/a004982f76256cab1b2d)