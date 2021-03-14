import pickle
from allennlp.data import (
    DataLoader,
    DatasetReader,
    Instance,
    Vocabulary,
    TextFieldTensors,
)
from allennlp.predictors import Predictor
from allennlp.common.util import JsonDict
from tqdm import tqdm
import os

class EmbeddingEncoder(Predictor):
    def predict(self, sentence: str) -> JsonDict:
        # This method is implemented in the base class.
        return self.predict_json({"context": sentence})

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        context = json_dict["context"]
        return self._dataset_reader.text_to_instance(mention_uniq_id=None,
                                                     data={'title': context})

class ArticleKB:
    def __init__(self, model, dsr, config):
        self.predictor = EmbeddingEncoder(model, dsr)
        self.dsr = dsr
        self.train_mention_ids, self.dev_mention_ids, self.mention_id2data = \
            dsr.train_mention_ids, dsr.dev_mention_ids, dsr.mention_id2data
        self.config = config

        self._dump_dir_maker()
        if not os.path.exists(self.config.dump_emb_dir+'kbemb.pkl'):
            mention_idx2emb = self._article_emb_iterator_from_train_and_dev_dataset()
            with open(self.config.dump_emb_dir+'kbemb.pkl', 'wb') as f:
                pickle.dump(mention_idx2emb, f)
            self.mention_idx2emb = mention_idx2emb
        else:
            with open(self.config.dump_emb_dir+'kbemb.pkl', 'rb') as g:
                self.mention_idx2emb = pickle.load(g)


    def _article_emb_iterator_from_train_and_dev_dataset(self):
        print('=== emb making from train and dev')

        mention_id2emb = {}

        for mention_id in tqdm(self.train_mention_ids + self.dev_mention_ids):
            its_article_title_emb = self.predictor.predict(
                self.mention_id2data[mention_id]['title']
            )['encoded_embeddings']
            mention_id2emb.update({mention_id: its_article_title_emb})

        return mention_id2emb

    def _dump_dir_maker(self):
        if not os.path.exists(self.config.dump_emb_dir):
            os.mkdir(self.config.dump_emb_dir)

