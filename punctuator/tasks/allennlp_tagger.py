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

from allennlp.data.token_indexers import PretrainedBertIndexer

from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
 
 # the token indexer is responsible for mapping tokens to integers
token_indexer = ELMoTokenCharactersIndexer()
#max_len = 100
#token_indexer = PretrainedBertIndexer(
#             pretrained_model="bert-base-uncased",
#                 max_pieces=max_len,
#                     do_lowercase=True,
#                      )
  
#def tokenizer(s: str):
#         return token_indexer.wordpiece_tokenizer(s)[:max_len - 2]
from allennlp.modules.token_embedders import ElmoTokenEmbedder
 
options_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json'
weight_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5'
elmo_embedder = ElmoTokenEmbedder(options_file, weight_file)

reader = PunctuatorDatasetReader(token_indexers={'tokens': token_indexer})
train_dataset = reader.read(str(PathManager.RAW / 'train.csv'))
dev_dataset = reader.read(str(PathManager.RAW / 'dev.csv'))
test_dataset = reader.read(str(PathManager.RAW / 'test.csv'))

vocab = Vocabulary.from_instances(train_dataset + dev_dataset)
token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                            embedding_dim=Config.EMBED_DIM)
# word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
word_embeddings = BasicTextFieldEmbedder({"tokens": elmo_embedder})

lstm = PytorchSeq2SeqWrapper(torch.nn.GRU(Config.EMBED_DIM, Config.HIDDEN_DIM, batch_first=True, bidirectional=True))
model: LstmTagger = LstmTagger(word_embeddings, lstm, vocab)

if torch.cuda.is_available():
    cuda_device = 0
    model = model.cuda(cuda_device)
else:
    cuda_device = -1

optimizer = optim.Adam(model.parameters())
iterator = BucketIterator(batch_size=Config.BATCH_SIZE, sorting_keys=[('sentence', 'num_tokens')])
iterator.index_with(vocab)
trainer = Trainer(model=model,
                  optimizer=optimizer,
                  iterator=iterator,
                  train_dataset=train_dataset,
                  validation_dataset=dev_dataset,
                  validation_metric='+accuracy',
                  patience=5,
                  summary_interval=5,
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
