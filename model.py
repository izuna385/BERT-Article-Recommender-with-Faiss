'''
Model classes
'''
import torch
import pdb
import torch.nn as nn
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper
from allennlp.models import Model
from overrides import overrides
from allennlp.training.metrics import CategoricalAccuracy, BooleanAccuracy
from torch.nn.functional import softmax

class TitleAndCaptionClassifier(Model):
    def __init__(self, args,
                 mention_encoder: Seq2VecEncoder,
                 num_label: int,
                 vocab):
        super().__init__(vocab)
        self.args = args
        self.mention_encoder = mention_encoder
        self.accuracy = BooleanAccuracy()
        self.BCEWloss = nn.BCEWithLogitsLoss()
        self.accuracy = CategoricalAccuracy()
        self.loss = nn.CrossEntropyLoss()
        self.linear_for_classify = nn.Linear(self.mention_encoder.get_output_dim(), num_label)

    def forward(self, context,
                mention_uniq_id: torch.Tensor = None,
                label: torch.Tensor = None):
        emb = self.mention_encoder(context)
        scores = self.linear_for_classify(emb)
        probs = softmax(scores, dim=1)
        output = {}
        if label is not None:
            loss = self.loss(scores, label)
            self.accuracy(probs, label)
            output['loss'] = loss
            output['logits'] = scores
            output['probs'] = probs
            output['mention_uniq_id'] = mention_uniq_id

        output['encoded_embeddings'] = emb
        return output

    @overrides
    def get_metrics(self, reset: bool = False):
        return {"accuracy": self.accuracy.get_metric(reset)}

