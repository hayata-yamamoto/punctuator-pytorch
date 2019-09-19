import numpy as np
import pandas as pd
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
    vocab = Vocabulary.from_files("/tmp/vocabulary")
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

    with open("/tmp/model.th", 'rb') as f:
        model.load_state_dict(torch.load(f))
    if cuda_device > -1:
        model.cuda(cuda_device)
    predictor = SentenceTaggerPredictor(model, dataset_reader=reader)

    s = """Machine learning (ML) is the scientific study of algorithms and statistical models that computer systems use to perform a specific task without using explicit instructions, relying on patterns and inference instead. It is seen as a subset of artificial intelligence. Machine learning algorithms build a mathematical model based on sample data, known as "training data", in order to make predictions or decisions without being explicitly programmed to perform the task.[1][2]:2 Machine learning algorithms are used in a wide variety of applications, such as email filtering and computer vision, where it is difficult or infeasible to develop a conventional algorithm for effectively performing the task."""
    labels = np.argmax(predictor.predict(s.replace('.', '').replace('?', '').replace('!', ''))['tag_logts'], axis=-1)
    #df = pd.read_csv(PathManager.INTERIM / 'val.csv')
    #labels = df.transcript.progress_apply(
    #    lambda s: np.argmax(predictor.predict(
    #        s.replace('.', '').replace('?', '').replace('!', '')
    #    )['tag_logits'], axis=-1)
    #)
    print(labels)


if __name__ == '__main__':
    main()
