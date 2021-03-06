'''
Model classes
'''
import torch
import torch.nn as nn
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper
from allennlp.models import Model
from overrides import overrides
from allennlp.training.metrics import CategoricalAccuracy, BooleanAccuracy

class TitleAndCaptionClassifier(Model):
    def __init__(self, args,
                 mention_encoder: Seq2VecEncoder,
                 vocab):
        super().__init__(vocab)
        self.args = args
        self.mention_encoder = mention_encoder
        self.accuracy = BooleanAccuracy()
        self.BCEWloss = nn.BCEWithLogitsLoss()
        self.mesloss = nn.MSELoss()
        self.istrainflag = 1
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.linear_for_classify = nn.Linear(self.mention_encoder.get_output_dim(), 1)

    def forward(self, context, label):
        scores = self.linear_for_classify(self.mention_encoder(context))

        loss = self.BCEWloss(scores.view(-1), label.float())
        output = {'loss': loss}

        if self.istrainflag:
            binary_class = (torch.sigmoid(scores.view(-1)) > 0.5).int()
            self.accuracy(binary_class, label)
        return output

    @overrides
    def get_metrics(self, reset: bool = False):
        return {"accuracy": self.accuracy.get_metric(reset)}

