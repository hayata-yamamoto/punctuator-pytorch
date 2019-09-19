import json

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import torch
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.predictors import SentenceTaggerPredictor

from punctuator.src.core.config import Config
from punctuator.src.core.path_manager import PathManager
from punctuator.src.datasets import TedDatasetReader
from punctuator.src.models import LstmTagger


def main():
    reader = TedDatasetReader()
    vocab = Vocabulary.from_files(str(PathManager.PROCESSED / 'vocabulary'))
    token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                embedding_dim=Config.EMBED_DIM)
    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
    lstm = PytorchSeq2SeqWrapper(
        torch.nn.GRU(Config.EMBED_DIM, Config.HIDDEN_DIM, batch_first=True, bidirectional=True))
    model: LstmTagger = LstmTagger(word_embeddings, lstm, vocab)

    if torch.cuda.is_available():
        cuda_device = 0
        model = model.cuda(cuda_device)
    else:
        cuda_device = -1

    with (PathManager.PROCESSED / "model.th").open(mode='rb') as f:
        # model.load_state_dict(torch.load(f))
        model.load_state_dict(torch.load(f, map_location='cpu'))
    if cuda_device > -1:
        model.cuda(cuda_device)
    predictor = SentenceTaggerPredictor(model, dataset_reader=reader)

    with (PathManager.PROCESSED / 'val.json').open('r') as f:
        df = pd.DataFrame.from_records(json.load(f))

    res = []
    for i, row in df.iterrows():
        logits = predictor.predict(s.lower().replace('.', '').replace('?', '').replace('!', ''))['tag_logits']
        ids = np.argmax(logits, axis=-1)
        labels = [model.vocab.get_token_from_index(i, 'labels') for i in ids]
        res.extend(labels)

    y_true = df.tags.values.flatten()
    print(classification_report(y_true, res))


if __name__ == '__main__':
    main()
