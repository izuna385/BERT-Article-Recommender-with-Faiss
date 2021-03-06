from dataset_reader import LivedoorCorpusReader
from parameters import Params
import pdb
from utils import build_vocab, build_data_loaders, build_trainer, emb_returner
from encoder import Pooler_for_mention
from model import TitleAndCaptionClassifier
from allennlp.training.util import evaluate
from emb_dumper import EmbeddingEncoder

if __name__ == '__main__':
    params = Params()
    config = params.opts
    dsr = LivedoorCorpusReader(config=config)

    # Loading Datasets
    train, dev, test = dsr._read('train'), dsr._read('dev'), dsr._read('test')
    train_and_dev = train + dev
    vocab = build_vocab(train_and_dev)
    num_label = len(dsr.class2id)
    train_loader, dev_loader, test_loader = build_data_loaders(config, train, dev, test)
    train_loader.index_with(vocab)
    dev_loader.index_with(vocab)

    _, __, embedder = emb_returner(config=config)
    mention_encoder = Pooler_for_mention(config, embedder)
    model = TitleAndCaptionClassifier(config, mention_encoder, num_label, vocab)
    trainer = build_trainer(config, model, train_loader, dev_loader)
    trainer.train()

    model.eval()
    test_loader.index_with(model.vocab)
    eval_result = evaluate(model=model,
                           data_loader=test_loader,
                           cuda_device=0,
                           batch_weight_key="")
    print(eval_result)

    embedding_encoder = EmbeddingEncoder(model, dsr)
    res = embedding_encoder.predict('test emb')