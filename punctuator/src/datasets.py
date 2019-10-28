from pathlib import Path
from typing import List, Iterable
from overrides import overrides
import pandas as pd
from allennlp.data import Instance, Token
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.dataset_readers import SequenceTaggingDatasetReader

from punctuator.src.core.path_manager import PathManager


class PunctuatorDatasetReader(SequenceTaggingDatasetReader):

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        df = pd.read_csv(file_path).dropna()

        for sent in df['0']:
            t = str(sent).strip().split(' ')
            tokens = [Token(s.split('###')[0]) for s in t]
            tags = [s.split('###')[1] for s in t]

            yield self.text_to_instance(tokens=tokens, tags=tags)

    def text_to_instance(self, tokens: List[Token], tags: List[str] = None) -> Instance:
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"sentence": sentence_field}

        if tags:
            label_field = SequenceLabelField(labels=tags, sequence_field=sentence_field)
            fields["labels"] = label_field

        return Instance(fields)


def ted_data() -> pd.DataFrame:
    return pd.read_csv(PathManager.RAW / 'transcripts.csv')


def write_txt(contents: List[str], filename: Path) -> None:
    with open(filename, 'w') as f:
        [f.write(s + '\n') for s in contents]


def tagmap(word: str) -> str:
    if word in [".", ";", "!"]:
        return "PERIOD"
    if word == "?":
        return "QUESTION"
    if word in [",", ":"]:
        return "COMMA"
    else:
        return "O"


def tagging(sentence: str) -> str:
    words = sentence\
        .replace('.', ' .')\
        .replace('?', ' ?')\
        .replace(',', ' ,')\
        .replace('!', ' !')\
        .replace(';', ' ;')\
        .replace(":", " :")\
        .split(' ')
    sent = []

    for i in range(len(words) - 1):
        w = words[i].replace('.', '').replace('?', '').replace('!', '').replace(',', '')
        if w == '':
            continue

        sent.append(f'{w}###{tagmap(words[i+1])}')
    return ' '.join(sent)


def make_data(dataset: List[str], filename: str) -> None:
    pd.DataFrame([tagging(d) for d in dataset]).to_csv(filename, index=False)
    # with open(filename, 'w') as f:
    #     for s in dataset:
    #         f.write(tagging(s) + '\n')
