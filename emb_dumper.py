from typing import Dict, Iterable, List, Tuple
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder

from allennlp.data import (
    DataLoader,
    DatasetReader,
    Instance,
    Vocabulary,
    TextFieldTensors,
)
from allennlp.data.data_loaders import SimpleDataLoader
from allennlp.models import Model
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
import pdb
from allennlp.predictors import Predictor
from allennlp.common.util import JsonDict
from allennlp.data.samplers import BucketBatchSampler

class EmbeddingEncoder(Predictor):
    def predict(self, sentence: str) -> JsonDict:
        # This method is implemented in the base class.
        return self.predict_json({"context": sentence})

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        context = json_dict["context"]
        return self._dataset_reader.text_to_instance(mention_uniq_id=None,
                                                     data={'title': context})

