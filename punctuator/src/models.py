from typing import Dict, Optional

import torch
import torch.nn.functional as F
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.nn.util import (get_text_field_mask, sequence_cross_entropy_with_logits)
from allennlp.training.metrics import CategoricalAccuracy


class Punctuator(Model):

    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 vocab: Vocabulary,
                 attention: Seq2SeqEncoder = None) -> None:
        super(Punctuator, self).__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.attention = attention
        self.hidden2tag = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                          out_features=vocab.get_vocab_size('labels'))
        self.accuracy = CategoricalAccuracy()

    def forward(self,
                sentence: Dict[str, torch.Tensor],
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(sentence)  # (batch_size, num_tokens)
        embeddings = self.word_embeddings(sentence)  # (batch_size,  num_rows, embedding_size)
        encoder_out = self.encoder(embeddings, mask)  # (batch_size, num_rows, hidden_size * 2)

        # if self.attention is not None:
        #     attn_out = self.attention(encoder_out, mask)
        #     encoder_out = encoder_out * attn_out

        tag_space = self.hidden2tag(encoder_out)
        tag_logits = F.log_softmax(tag_space, dim=1)
        output = {"tag_logits": tag_logits}

        if labels is not None:
            self.accuracy(tag_logits, labels, mask)
            output["loss"] = sequence_cross_entropy_with_logits(tag_logits, labels, mask)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            'accuracy': self.accuracy.get_metric(reset),
        }
