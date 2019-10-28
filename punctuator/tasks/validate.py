import pandas as pd
import numpy as np
import torch
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.predictors import SentenceTaggerPredictor
from sklearn.metrics import classification_report
from tqdm import tqdm

from punctuator.src.core.config import Config
from punctuator.src.core.path_manager import PathManager
from punctuator.src.datasets import PunctuatorDatasetReader
from punctuator.src.models import LstmTagger


def main():
    reader = PunctuatorDatasetReader()
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

    df = pd.read_csv(PathManager.RAW / 'test.csv')

    pred = []
    true = []
    s = df['0']
    print(preditor.predict(str(s)))
#    for s in tqdm(df['0']):
#        logit = predictor.predict(str(s))['tag_logits']
#        idx = np.argmax(logit[0], axis=-1)
#        pred.append(model.vocab.get_token_from_index(idx, 'labels'))
#        true.append(t)

#    print(classification_report(true, pred))


if __name__ == '__main__':
    main()
