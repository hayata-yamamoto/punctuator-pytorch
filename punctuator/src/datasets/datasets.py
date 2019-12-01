from typing import Dict, Iterable, List

import pandas as pd
from allennlp.data import Instance, Token
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import SequenceLabelField, TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from overrides import overrides


class PunctuatorTokenizer:

    @staticmethod
    def split_words(s):
        return [Token(_.split("###")[0]) for _ in str(s).split(" ")]


class PunctuatorDatasetReader(DatasetReader):

    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        df = pd.read_csv(file_path).dropna()

        for sent in df["0"]:
            t = str(sent).strip().split(" ")
            tokens = [Token(s.split("###")[0]) for s in t]
            tags = [s.split("###")[1] for s in t]

            yield self.text_to_instance(tokens=tokens, tags=tags)

    def text_to_instance(self, tokens: List[Token], tags: List[str] = None) -> Instance:
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"sentence": sentence_field}

        if tags:
            label_field = SequenceLabelField(labels=tags, sequence_field=sentence_field)
            fields["labels"] = label_field

        return Instance(fields)
