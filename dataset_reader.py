import pdb
from overrides import overrides
from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import SpanField, ListField, TextField, MetadataField, ArrayField, SequenceLabelField, LabelField
from allennlp.data.fields import LabelField, TextField
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer
from parameters import Params
import random
from tqdm import tqdm
from tokenizer import CustomTokenizer
import numpy as np
import glob
import re
import math
random.seed(42)

class LivedoorCorpusReader(DatasetReader):
    def __init__(
        self,
        config,
        max_tokens: int = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.custom_tokenizer_class = CustomTokenizer(config=config)
        self.token_indexers = self.custom_tokenizer_class.token_indexer_returner()
        self.max_tokens = max_tokens
        self.config = config
        self.train_mention_ids, self.dev_mention_ids, self.test_mention_ids, self.mention_id2data = \
            self._mention_ids_returner()

        self.class2id = {}

    @overrides
    def _read(self, train_dev_test_flag: str) -> list:
        '''
        :param train_dev_test_flag: 'train', 'dev', 'test'
        :return: list of instances
        '''
        mention_ids, instances = list(), list()
        if train_dev_test_flag == 'train':
            mention_ids += self.train_mention_ids
            # Because Iterator(shuffle=True) has bug, we forcefully shuffle train dataset here.
            random.shuffle(mention_ids)
        elif train_dev_test_flag == 'dev':
            mention_ids += self.dev_mention_ids
        elif train_dev_test_flag == 'test':
            mention_ids += self.test_mention_ids

        if self.config.debug:
            mention_ids = mention_ids[:50]

        for idx, mention_uniq_id in tqdm(enumerate(mention_ids)):
            instances.append(self.text_to_instance(mention_uniq_id, data=self.mention_id2data[mention_uniq_id]))

        return instances

    @overrides
    def text_to_instance(self, mention_uniq_id, data=None) -> Instance:
        title_tokenized = [Token('[CLS]')]
        title_tokenized += [Token(split_token) for split_token in self.custom_tokenizer_class.tokenize(txt=data['title'])][:self.config.max_title_length]
        title_tokenized.append(Token('[unused1]'))
        title_tokenized += [Token(split_token) for split_token in self.custom_tokenizer_class.tokenize(txt=data['caption'])][:self.config.max_caption_length]
        title_tokenized += [Token('[SEP]')]

        context_field = TextField(title_tokenized, self.token_indexers)
        fields = {"context": context_field}

        if data['class'] not in self.class2id:
            self.class2id.update({data['class']: len(self.class2id)})

        fields['label'] = ArrayField(np.array(self.class2id[data['class']]))
        # fields['mention_uniq_id'] = ArrayField(np.array(mention_uniq_id))

        return Instance(fields)


    def _mention_ids_returner(self):
        mention_id2data = {}
        train_mention_ids, dev_mention_ids, test_mention_ids = [], [], []
        dataset_each_class_dirs_path = glob.glob(self.config.dataset_dir+'**/')

        for each_class_dir_path in dataset_each_class_dirs_path:
            file_paths = self._each_class_dir_path2_txt_paths(each_class_dir_path)
            random.shuffle(file_paths)
            # train : dev : test = 7 : 1 : 2
            data_num_in_one_label = len(file_paths)
            data_frac = data_num_in_one_label // 10
            train_tmp_ids = [i for i in range(0, data_frac * 7)]
            dev_tmp_ids = [j for j in range(data_frac * 7, data_frac * 8)]
            test_tmp_ids = [k for k in range(data_frac * 8, data_num_in_one_label)]

            for idx, file_path in enumerate(file_paths):
                data = self._one_file_reader(file_path)
                tmp_idx_for_all_data = len(mention_id2data)
                mention_id2data.update({tmp_idx_for_all_data: data})

                if idx in train_tmp_ids:
                    train_mention_ids.append(idx)
                elif idx in dev_tmp_ids:
                    dev_mention_ids.append(idx)
                elif idx in test_tmp_ids:
                    test_mention_ids.append(idx)
                else:
                    print('Error')
                    exit()

        return train_mention_ids, dev_mention_ids, test_mention_ids, mention_id2data

    def _each_class_dir_path2_txt_paths(self, each_class_dir_path):
        return [path for path in glob.glob(each_class_dir_path+'*') if 'LICENSE' not in path]

    def _one_file_reader(self, filepath):
        data = {'class': re.search(r'(.+)\/', filepath.replace(self.config.dataset_dir,'')).group(1)}

        with open(filepath, 'r') as f:
            for idx, line in enumerate(f):
                if idx == 2:
                    data.update({'title': line.strip()})
                if idx == 3:
                    if line.strip() == '':
                        continue
                    data.update({'caption': line.strip()})
                if idx > 3 and 'caption' not in data:
                    if line.strip() != '':
                        data.update({'caption': line.strip()})

        return data

if __name__ == '__main__':
    params = Params()
    config = params.opts
    dsr = LivedoorCorpusReader(config=config)
    dsr._read('train')