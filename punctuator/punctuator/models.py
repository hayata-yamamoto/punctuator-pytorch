from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.nn.util import (get_text_field_mask,
                              sequence_cross_entropy_with_logits)
from allennlp.training.metrics import CategoricalAccuracy


class Attn(nn.Module):
    # TODO: depreciate in future
    def __init__(self, input_sz: int, nch: int = 24) -> None:
        super(Attn, self).__init__()
        self.input_sz = input_sz
        self.main = nn.Sequential(nn.Linear(input_sz, nch), nn.ReLU(True),
                                  nn.Linear(nch, 1))

    def forward(
        self,
        encoder_outputs: torch.
        Tensor  # (batch_size, seq_len, hidden_sz(=input_sz))
    ):
        b_size = encoder_outputs.size(0)
        attn_ene = self.main(encoder_outputs.view(
            -1, self.input_sz))  # (b, s, h) -> (b*s, 1)
        return F.softmax(attn_ene.view(b_size, -1),
                         dim=1).unsqueeze(2)  # (b*s, 1) -> (b, s, 1)


class Punctuator(Model):
    def __init__(self, word_embeddings: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder, vocab: Vocabulary) -> None:
        super(Punctuator, self).__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.attention = Attn(self.encoder.get_output_dim())
        self.hidden2tag = torch.nn.Linear(
            in_features=encoder.get_output_dim(),
            out_features=vocab.get_vocab_size('labels'))
        self.accuracy = CategoricalAccuracy()

    def forward(
            self,
            sentence: Dict[str, torch.Tensor],
            labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(sentence)  # (batch_size, num_tokens)
        emb = self.word_embeddings(
            sentence)  # (batch_size, num_rows, embedding_size)
        out = self.encoder(emb,
                           mask)  # (batch_size, num_rows, hidden_size * 2)

        # TODO: remove comment out
        # attn = self.attention(out)
        # out = (out * attn)

        tag_logits = self.hidden2tag(out)
        output = {"tag_logits": tag_logits}

        if labels is not None:
            self.accuracy(tag_logits, labels, mask)
            output["loss"] = sequence_cross_entropy_with_logits(
                tag_logits, labels, mask)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            'accuracy': self.accuracy.get_metric(reset),
        }
