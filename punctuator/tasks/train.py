import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from allennlp.data.iterators import BucketIterator
from allennlp.data.token_indexers.elmo_indexer import \
    ELMoTokenCharactersIndexer
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import ElmoTokenEmbedder
from allennlp.predictors import SentenceTaggerPredictor
from allennlp.training import Trainer
from sklearn.metrics import classification_report
from comet_ml import Experiment

from punctuator.src.config import Config
from punctuator.src.datasets import (PunctuatorDatasetReader,
                                     PunctuatorTokenizer)
from punctuator.src.models import Punctuator
from punctuator.src.path_manager import PathManager


def main():
    experiment = Experiment(api_key=Config.CometMl.API_KEY,
                            project_name=Config.CometMl.PROJECT_NAME,
                            workspace=Config.CometMl.WORKSPACE)

    options_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json'
    weight_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5'

    token_indexer = ELMoTokenCharactersIndexer()
    reader = PunctuatorDatasetReader(token_indexers={'tokens': token_indexer})
    train_dataset = reader.read(str(PathManager.RAW / 'train.csv'))
    dev_dataset = reader.read(str(PathManager.RAW / 'dev.csv'))

    vocab = Vocabulary.from_instances(train_dataset + dev_dataset)
    # token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
    # embedding_dim=Config.EMBED_DIM)
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

    optimizer = optim.Adam(model.parameters())
    iterator = BucketIterator(batch_size=Config.BATCH_SIZE,
                              sorting_keys=[('sentence', 'num_tokens')])
    iterator.index_with(vocab)
    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=train_dataset,
                      validation_dataset=dev_dataset,
                      validation_metric='+f1',
                      patience=10,
                      summary_interval=10,
                      num_epochs=Config.EPOCH,
                      cuda_device=cuda_device)
    trainer.train()

    # Here's how to save the model.
    with (PathManager.PROCESSED / "model.th").open(mode='wb') as f:
        torch.save(model.state_dict(), f)
    vocab.save_to_files(PathManager.PROCESSED / "vocabulary")

    predictor = SentenceTaggerPredictor(model, dataset_reader=reader)
    predictor._tokenizer = PunctuatorTokenizer()
    df = pd.read_csv(PathManager.RAW / 'test.csv')

    pred = []
    true = []
    for s in tqdm(df['0']):
        logit = predictor.predict(str(s))['tag_logits']
        idx = [np.argmax(logit[i], axis=-1) for i in range(len(logit))]
        pred += [model.vocab.get_token_from_index(i, 'labels') for i in idx]
        true += [_.split('###')[1] for _ in s.split(' ')]

    print(classification_report(true, pred))

    s = """
Machine learning (ML) is the scientific study of algorithms and statistical models 
that computer systems use to perform a specific task without using explicit 
instructions, relying on patterns and inference instead. 
It is seen as a subset of artificial intelligence. 
Machine learning algorithms build a mathematical model based on sample data, known as
 "training data", in order to make predictions or decisions without being explicitly
  programmed to perform the task.[1][2]:2 Machine learning algorithms are used in a wide 
  variety of applications, such as email filtering and computer vision, where it is difficult or infeasible
   to develop a conventional algorithm for effectively performing the task.
"""
    logit = predictor.predict(str(s))['tag_logits']
    idx = [np.argmax(logit[i], axis=-1) for i in range(len(logit))]
    pred += [model.vocab.get_token_from_index(i, 'labels') for i in idx]

    def reconstruct(seq: List[str], labels: List[str]) -> str:
        res = []
        for s, l in zip(seq, labels):
            if l == "PERIOD":
                res.append(s + ".")
            elif l == "QUESTION":
                res.append(s + "?")
            elif l == "COMMA":
                res.append(s + ",")
            else:
                res.append(s)
        return " ".join(res)

    print("==============")
    print('True Sentence ')
    print("==============")
    print(s)

    print("==============")
    print('Predict Sentence ')
    print("==============")
    print(reconstruct(s.strip().split(' '), pred))


if __name__ == '__main__':
    main()
