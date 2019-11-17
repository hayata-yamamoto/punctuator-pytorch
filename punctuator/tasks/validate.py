import numpy as np
import torch
from allennlp.data.token_indexers.elmo_indexer import \
    ELMoTokenCharactersIndexer
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import ElmoTokenEmbedder
from allennlp.predictors import SentenceTaggerPredictor

from punctuator.src.config import Config
from punctuator.src.datasets import (PunctuatorDatasetReader,
                                     PunctuatorTokenizer)
from punctuator.src.models import Punctuator
from punctuator.src.path_manager import PathManager


def main():

    options_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json'
    weight_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5'

    token_indexer = ELMoTokenCharactersIndexer()

    reader = PunctuatorDatasetReader(token_indexers={'tokens': token_indexer})
    vocab = Vocabulary.from_files(str(PathManager.PROCESSED / 'vocabulary'))
    token_embedding = ElmoTokenEmbedder(options_file, weight_file)
    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})

    lstm = PytorchSeq2SeqWrapper(
        torch.nn.GRU(Config.EMBED_DIM,
                     Config.HIDDEN_DIM,
                     batch_first=True,
                     bidirectional=True))
    model: Punctuator = Punctuator(word_embeddings, lstm, vocab)

    if torch.cuda.is_available():
        cuda_device = 0
        model = model.cuda(cuda_device)
    else:
        cuda_device = -1

    with (PathManager.PROCESSED / "model.th").open(mode='rb') as f:
        model.load_state_dict(torch.load(f))
        # model.load_state_dict(torch.load(f, map_location='cpu'))
    if cuda_device > -1:
        model.cuda(cuda_device)
    predictor = SentenceTaggerPredictor(model, dataset_reader=reader)
    predictor._tokenizer = PunctuatorTokenizer()

    while True:
        sentence = input()
        out = ''
        for s in sentence.split(' '):
            logit = predictor.predict(s)['tag_logits']
            idx = np.argmax(logit, axis=-1)
            label = model.vocab.get_token_from_index(idx, 'labels')


if __name__ == '__main__':
    main()
