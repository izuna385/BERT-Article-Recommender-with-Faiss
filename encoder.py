'''
Seq2VecEncoders for encoding mentions and entities.
'''
import torch.nn as nn
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper, BagOfEmbeddingsEncoder
from allennlp.modules.seq2vec_encoders import BertPooler
from overrides import overrides
from allennlp.nn.util import get_text_field_mask
from allennlp.modules.seq2vec_encoders.cls_pooler import ClsPooler

class Pooler_for_mention(Seq2VecEncoder):
    def __init__(self, args, word_embedder):
        super(Pooler_for_mention, self).__init__()
        self.args = args
        self.huggingface_nameloader()
        self.bertpooler_sec2vec = BertPooler(pretrained_model=self.bert_weight_filepath)
        self.word_embedder = word_embedder
        self.word_embedding_dropout = nn.Dropout(self.args.word_embedding_dropout)

    def huggingface_nameloader(self):
        if self.args.bert_name == 'japanese-bert':
            self.bert_weight_filepath = 'cl-tohoku/bert-base-japanese'
        else:
            self.bert_weight_filepath = 'dummy'
            print('Currently not supported', self.args.bert_name)
            exit()

    def forward(self, contextualized_mention):
        mask_sent = get_text_field_mask(contextualized_mention)
        mention_emb = self.word_embedder(contextualized_mention)
        mention_emb = self.word_embedding_dropout(mention_emb)
        mention_emb = self.bertpooler_sec2vec(mention_emb, mask_sent)

        return mention_emb

    @overrides
    def get_output_dim(self):
        return 768