import numpy as np
import torch
import torch.optim as optim
from allennlp.data.iterators import BucketIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.predictors import SentenceTaggerPredictor
from allennlp.training import Trainer

from punctuator.src.core.config import Config
from punctuator.src.core.path_manager import PathManager
from punctuator.src.datasets import PunctuatorDatasetReader
from punctuator.src.models import LstmTagger

reader = PunctuatorDatasetReader()
train_dataset = reader.read(str(PathManager.RAW / 'train.csv'))
dev_dataset = reader.read(str(PathManager.RAW / 'dev.csv'))
test_dataset = reader.read(str(PathManager.RAW / 'test.csv'))

vocab = Vocabulary.from_instances(train_dataset + dev_dataset)

token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                            embedding_dim=Config.EMBED_DIM)
word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
lstm = PytorchSeq2SeqWrapper(torch.nn.GRU(Config.EMBED_DIM, Config.HIDDEN_DIM, batch_first=True, bidirectional=True))
model: LstmTagger = LstmTagger(word_embeddings, lstm, vocab)

if torch.cuda.is_available():
    cuda_device = 0
    model = model.cuda(cuda_device)
else:
    cuda_device = -1

optimizer = optim.Adam(model.parameters(), lr=Config.LR)
iterator = BucketIterator(batch_size=Config.BATCH_SIZE, sorting_keys=[('sentence', 'num_tokens')])
iterator.index_with(vocab)
trainer = Trainer(model=model,
                  optimizer=optimizer,
                  iterator=iterator,
                  train_dataset=train_dataset,
                  validation_dataset=dev_dataset,
                  validation_metric='+accuracy',
                  patience=10,
                  summary_interval=10,
                  num_epochs=Config.EPOCH,
                  cuda_device=cuda_device)
trainer.train()

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

predictor = SentenceTaggerPredictor(model, dataset_reader=reader)
tag_logits = predictor.predict(s)['tag_logits']
tag_ids = np.argmax(tag_logits, axis=-1)
print([model.vocab.get_token_from_index(i, 'labels') for i in tag_ids])

# Here's how to save the model.
with (PathManager.PROCESSED / "model.th").open(mode='wb') as f:
    torch.save(model.state_dict(), f)
vocab.save_to_files(PathManager.PROCESSED / "vocabulary")
# And here's how to reload the model.
