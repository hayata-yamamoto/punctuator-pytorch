import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from allennlp.data.iterators import BucketIterator
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import ElmoTokenEmbedder
from allennlp.predictors import SentenceTaggerPredictor
from allennlp.training import Trainer
from sklearn.metrics import classification_report
from tqdm import tqdm

from punctuator.src.config import Config, EnvFile
from punctuator.src.datasets.datasets import (
    PunctuatorDatasetReader,
    PunctuatorTokenizer,
)
from punctuator.src.models import Punctuator
from punctuator.src.path_manager import PathManager


def main():
    torch.manual_seed(1)
    token_indexer = ELMoTokenCharactersIndexer()
    reader = PunctuatorDatasetReader(token_indexers={"tokens": token_indexer})

    vocab = Vocabulary.from_files(str(PathManager.PROCESSED / 'vocabulary'))
    token_embedding = ElmoTokenEmbedder(EnvFile.OPTIONS_FILE, EnvFile.WEIGHT_FILE)
    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})

    lstm = PytorchSeq2SeqWrapper(
        torch.nn.GRU(Config.EMBED_DIM, Config.HIDDEN_DIM, batch_first=True, bidirectional=True))
    model = Punctuator(word_embeddings, lstm, vocab)

    with (PathManager.PROCESSED / 'elmo_model.th').open(mode='rb') as f:
        model.load_state_dict(torch.load(f))

    if torch.cuda.is_available():
        cuda_device = 0
        model = model.cuda(cuda_device)
    else:
        cuda_device = -1

    predictor = SentenceTaggerPredictor(model, dataset_reader=reader)
    predictor._tokenizer = PunctuatorTokenizer()


    s = """Machine learning (ML) is the scientific study of algorithms and statistical models that computer systems use to perform a specific task without using explicit instructions, relying on patterns and inference instead. It is seen as a subset of artificial intelligence. Machine learning algorithms build a mathematical model based on sample data, known as "training data", in order to make predictions or decisions without being explicitly programmed to perform the task.[1][2]:2 Machine learning algorithms are used in a wide variety of applications, such as email filtering and computer vision, where it is difficult or infeasible to develop a conventional algorithm for effectively performing the task."""
    logit = predictor.predict(str(s))["tag_logits"]
    idx = [np.argmax(logit[i], axis=-1) for i in range(len(logit))]
    pred += [model.vocab.get_token_from_index(i, "labels") for i in idx]

    print("==============")
    print("True Sentence ")
    print("==============")
    print(s)

    print("==============")
    print("Predict Sentence ")
    print("==============")
    print(reconstruct(s.strip().split(" "), pred))


if __name__ == "__main__":
    main()
