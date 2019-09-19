import numpy as np
from argparse import ArgumentParser
import torch
import torch.optim as optim
from allennlp.data.iterators import BucketIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.predictors import SentenceTaggerPredictor
from allennlp.training import Trainer

from punctuator.src.core.path_manager import PathManager
from punctuator.src.datasets import TedDatasetReader
from punctuator.src.models import LstmTagger


parser = ArgumentParser()
parser.add_argument('--embed', default=100, type=int)
parser.add_argument('--hidden', default=100, type=int)
parser.add_argument('--epoch', default=100, type=int)
parser.add_argument('--batch', default=2, type=int)

args = parser.parse_args()

reader = TedDatasetReader()
train_dataset = reader.read(str(PathManager.PROCESSED / 'train.txt'))
validation_dataset = reader.read(str(PathManager.PROCESSED / 'val.txt'))
vocab = Vocabulary.from_instances(train_dataset + validation_dataset)

token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                            embedding_dim=args.embed)
word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(args.embed, args.hidden, batch_first=True))
model: LstmTagger = LstmTagger(word_embeddings, lstm, vocab)

if torch.cuda.is_available():
    cuda_device = 0
    model = model.cuda(cuda_device)
else:
    cuda_device = -1

optimizer = optim.Adam(model.parameters(), lr=0.1)
iterator = BucketIterator(batch_size=args.batch, sorting_keys=[("sentence", "num_tokens")])
iterator.index_with(vocab)
trainer = Trainer(model=model,
                  optimizer=optimizer,
                  iterator=iterator,
                  train_dataset=train_dataset,
                  validation_dataset=validation_dataset,
                  patience=10,
                  summary_interval=10,
                  num_epochs=args.epoch,
                  cuda_device=cuda_device)
trainer.train()
predictor = SentenceTaggerPredictor(model, dataset_reader=reader)
tag_logits = predictor.predict("The dog ate the apple")['tag_logits']
tag_ids = np.argmax(tag_logits, axis=-1)
print([model.vocab.get_token_from_index(i, 'labels') for i in tag_ids])

# Here's how to save the model.
with open("/tmp/model.th", 'wb') as f:
    torch.save(model.state_dict(), f)
vocab.save_to_files("/tmp/vocabulary")
# And here's how to reload the model.
# vocab2 = Vocabulary.from_files("/tmp/vocabulary")
# model2: LstmTagger = LstmTagger(word_embeddings, lstm, vocab2)
# with open("/tmp/model.th", 'rb') as f:
#     model2.load_state_dict(torch.load(f))
# if cuda_device > -1:
#     model2.cuda(cuda_device)
# predictor2 = SentenceTaggerPredictor(model2, dataset_reader=reader)
# tag_logits2 = predictor2.predict("The dog ate the apple")['tag_logits']
# np.testing.assert_array_almost_equal(tag_logits2, tag_logits)