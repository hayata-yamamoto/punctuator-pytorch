import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from allennlp.data.iterators import BucketIterator
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper, IntraSentenceAttentionEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.predictors import SentenceTaggerPredictor
from allennlp.training import Trainer
from sklearn.metrics import classification_report
from allennlp.common.params import Params
from tqdm import tqdm

from punctuator.src.config import Config
from punctuator.src.datasets import (
    PunctuatorDatasetReader,
    PunctuatorTokenizer,
)
from punctuator.src.utils import replacing
from punctuator.src.models import Punctuator
from punctuator.src.path_manager import PathManager


def main():
    torch.manual_seed(1)
    token_indexer = SingleIdTokenIndexer()
    reader = PunctuatorDatasetReader(token_indexers={"tokens": token_indexer})
    train_dataset = reader.read(str(PathManager.RAW / "train.csv"))
    dev_dataset = reader.read(str(PathManager.RAW / "dev.csv"))

    vocab = Vocabulary.from_instances(train_dataset + dev_dataset)
    token_embedding = Embedding.from_params(
        vocab,
        Params({
            "embedding_dim": Config.GLOVE_DIM,
            "pretrained_file": '(https://nlp.stanford.edu/data/glove.twitter.27B.zip)#glove.twitter.27B.200d.txt'
        }))
    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})

    gru = PytorchSeq2SeqWrapper(torch.nn.GRU(Config.GLOVE_DIM, Config.HIDDEN_DIM, batch_first=True,
                                             bidirectional=True))
    attn = IntraSentenceAttentionEncoder(input_dim=Config.GLOVE_DIM,
                                         projection_dim=Config.HIDDEN_DIM * 2,
                                         combination='2')
    model: Punctuator = Punctuator(
        word_embeddings,
        gru,
        vocab,
        # attn
    )

    if torch.cuda.is_available():
        cuda_device = 0
        model = model.cuda(cuda_device)
    else:
        cuda_device = -1

    optimizer = optim.Adam(model.parameters(), lr=Config.LR)
    iterator = BucketIterator(batch_size=Config.BATCH_SIZE, sorting_keys=[("sentence", "num_tokens")])
    iterator.index_with(vocab)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        iterator=iterator,
        train_dataset=train_dataset,
        validation_dataset=dev_dataset,
        validation_metric="-loss",
        patience=Config.PATIENCE,
        summary_interval=Config.SUMMARY_INTERVAL,
        num_epochs=1,
        cuda_device=cuda_device,
    )

    trainer.train()

    # Here's how to save the model.
    with (PathManager.PROCESSED / "glove_model.th").open(mode="wb") as f:
        torch.save(model.state_dict(), f)
    vocab.save_to_files(str(PathManager.PROCESSED / "vocabulary"))

    predictor = SentenceTaggerPredictor(model, dataset_reader=reader)
    predictor._tokenizer = PunctuatorTokenizer()
    df = pd.read_csv(PathManager.RAW / "test.csv")

    pred = []
    true = []
    for s in tqdm(df["0"]):
        sent = replacing(str(s))
        logit = predictor.predict(sent)["tag_logits"]
        idx = [np.argmax(logit[i], axis=-1) for i in range(len(logit))]
        pred += [model.vocab.get_token_from_index(i, "labels") for i in idx]
        true += [_.split("###")[1] for _ in sent.split(" ")]

    print(classification_report(true, pred))


if __name__ == "__main__":
    main()