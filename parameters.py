import argparse
import sys, json
from distutils.util import strtobool
import pdb

class Params:
    def __init__(self):
        parser = argparse.ArgumentParser(description='NED')
        parser.add_argument('-debug', action='store', default=False, type=strtobool)
        parser.add_argument('-dataset_dir', action="store", default="./dataset/text/", type=str)
        parser.add_argument('-serialization_dir', action="store", default="./serialization_dir/", type=str)
        parser.add_argument('-dump_emb_dir', action="store", default="./dump_emb_dir/", type=str)
        parser.add_argument('-bert-name', action="store", default="japanese-bert", type=str)

        parser.add_argument('-lr', action="store", default=1e-5, type=float)
        parser.add_argument('-weight_decay', action="store", default=0, type=float)
        parser.add_argument('-beta1', action="store", default=0.9, type=float)
        parser.add_argument('-beta2', action="store", default=0.999, type=float)
        parser.add_argument('-epsilon', action="store", default=1e-8, type=float)
        parser.add_argument('-amsgrad', action='store', default=False, type=strtobool)
        parser.add_argument('-word_embedding_dropout', action="store", default=0.0, type=float)
        parser.add_argument('-cuda_devices', action="store", default='0', type=str)
        parser.add_argument('-num_epochs', action="store", default=5, type=int)

        parser.add_argument('-batch_size_for_train', action="store", default=64, type=int)
        parser.add_argument('-batch_size_for_eval', action="store", default=64, type=int)
        parser.add_argument('-debug_sample_num', action="store", default=2000, type=int)
        parser.add_argument('-max_title_length', action="store", default=30, type=int)
        parser.add_argument('-max_caption_length', action="store", default=70, type=int)
        parser.add_argument('-max_token_length', action="store", default=128, type=int)

        parser.add_argument('-search_method_for_faiss', action="store", default='indexflatl2', type=str)
        parser.add_argument('-how_many_top_hits_preserved', action="store", default=5, type=int)

        self.all_opts = parser.parse_known_args(sys.argv[1:])
        self.opts = self.all_opts[0]
        print('\n===PARAMETERS===')
        for arg in vars(self.all_opts[0]):
            print(arg, getattr(self.opts, arg))
        print('===PARAMETERS END===\n')

    def get_params(self):
        return self.opts

    def dump_params(self, experiment_dir):
        parameters = vars(self.get_params())
        with open(experiment_dir + 'parameters.json', 'w') as f:
            json.dump(parameters, f, ensure_ascii=False, indent=4, sort_keys=False, separators=(',', ': '))