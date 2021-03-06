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
        hidden1_dim = self.mention_encoder.get_output_dim() // 2
        hidden2_dim = hidden1_dim // 2
        hidden3_dim = hidden2_dim // 2
        self.fc = nn.Sequential(
            nn.Linear(self.mention_encoder.get_output_dim(), hidden1_dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden1_dim, hidden2_dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden2_dim, hidden3_dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden3_dim, num_label))

    def forward(self, context, label):

        scores = self.linear_for_classify(self.mention_encoder(context))
        probs = softmax(scores, dim=1)
        loss = self.loss(scores, label)
        output = {'loss': loss}
        output['logits'] = scores
        output['probs'] = probs
        self.accuracy(probs, label)

        return output

    @overrides
    def get_metrics(self, reset: bool = False):
        return {"accuracy": self.accuracy.get_metric(reset)}

