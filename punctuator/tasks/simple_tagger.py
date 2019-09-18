from flair.datasets import ColumnCorpus
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.embeddings import WordEmbeddings, StackedEmbeddings
from punctuator.src.core.path_manager import PathManager


cols = {0: 'word', 1: 'PUNCT'}
corpus = ColumnCorpus(PathManager.PROCESSED, cols, train_file='train.txt')

tag_type = 'PUNCT'
dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
print(dictionary.idx2item)

embedding_types = [
    WordEmbeddings('glove')
]

embeddings = StackedEmbeddings(embeddings=embedding_types)
tagger = SequenceTagger(hidden_size=256, embeddings=embeddings, tag_dictionary=dictionary, tag_type=tag_type)
trainer = ModelTrainer(tagger, corpus, use_tensorboard=True)

trainer.train(PathManager.INTERIM / 'punctuator', max_epochs=10, mini_batch_size=10, eval_mini_batch_size=10, monitor_test=True, monitor_train=True)

# plotter = Plotter()
# plotter.plot_weights(PathManager.INTERIM / 'punctuator/weights.txt')
