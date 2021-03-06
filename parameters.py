import argparse
import sys, json
from distutils.util import strtobool

class Params:
    def __init__(self):
        parser = argparse.ArgumentParser(description='NED')
        parser.add_argument('-debug', action='store', default=False, type=strtobool)
        parser.add_argument('-dataset_dir', action="store", default="./dataset/text/", type=str)
        parser.add_argument('-bert-name', action="store", default="japanese-bert", type=str)

        parser.add_argument('-lr', action="store", default=5e-5, type=float)
        parser.add_argument('-weight_decay', action="store", default=0, type=float)
        parser.add_argument('-beta1', action="store", default=0.9, type=float)
        parser.add_argument('-beta2', action="store", default=0.999, type=float)
        parser.add_argument('-epsilon', action="store", default=1e-8, type=float)
        parser.add_argument('-amsgrad', action='store', default=False, type=strtobool)
        parser.add_argument('-word_embedding_dropout', action="store", default=0.1, type=float)
        parser.add_argument('-cuda_devices', action="store", default='0', type=str)
        parser.add_argument('-num_epochs', action="store", default=5, type=int)

        parser.add_argument('-batch_size_for_train', action="store", default=16, type=int)
        parser.add_argument('-batch_size_for_eval', action="store", default=16, type=int)
        parser.add_argument('-debug_sample_num', action="store", default=2000, type=int)
        parser.add_argument('-max_title_length', action="store", default=30, type=int)
        parser.add_argument('-max_caption_length', action="store", default=250, type=int)

        self.opts = parser.parse_args(sys.argv[1:])
        print('\n===PARAMETERS===')
        for arg in vars(self.opts):
            print(arg, getattr(self.opts, arg))
        print('===PARAMETERS END===\n')

    def get_params(self):
        return self.opts

    def dump_params(self, experiment_dir):
        parameters = vars(self.get_params())
        with open(experiment_dir + 'parameters.json', 'w') as f:
            json.dump(parameters, f, ensure_ascii=False, indent=4, sort_keys=False, separators=(',', ': '))