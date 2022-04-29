import os
from pprint import pprint
from typing import *

import torch
from pytorch_lightning.utilities import AttributeDict
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchtext.vocab import Vocab

from stud.constants import PAD_INDEX
from stud.dataset import ABSADataModule
from stud.layers import BertEmbedding, Attention, TransformerEncoder


class HParams:

    def __init__(self, dm: ABSADataModule, models_dir: str, mode: str) -> None:
        bert_model_name_or_path = (f"{models_dir}bert-base-cased"
                                   if os.path.isdir(f"{models_dir}bert-base-cased")
                                   else "bert-base-cased")

        num_classes = (len(dm.vocabs["tag"])
                       if mode != "cd"
                       else len(dm.vocabs["categories"]) * (len(dm.vocabs["category_polarities"]) + 1))

        self.vocab_size: int = len(dm.vocabs["text"])
        self.hidden_dim: int = 128
        self.embedding_dim: int = 300
        self.pos_embedding: bool = False
        self.pos_embedding_dim: int = 120
        self.pos_vocab_size: int = len(dm.vocabs["pos"])
        self.bert_embedding: bool = True
        self.bert_finetuning: bool = False
        self.bert_layers_to_merge: Sequence[int] = [-1, -2, -3, -4]
        self.bert_layer_pooling_strategy: str = "mean" if mode == "cd" else "second_to_last"
        self.bert_wordpiece_pooling_strategy: str = "mean"
        self.bert_model_name_or_path: str = bert_model_name_or_path
        self.pack_lstm_input: bool = True
        self.label_vocab: Vocab = dm.vocabs["tag" if mode != "cd" else "categories"]
        self.polarity_vocab: Vocab = dm.vocabs["category_polarities"] if mode == "cd" else None
        self.num_classes: int = num_classes
        self.bidirectional: bool = True
        self.num_layers: int = 2
        self.dropout: float = 0.5
        self.max_epochs: int = 150
        self.attention: bool = mode == "cd"
        self.attention_heads: int = 12
        self.attention_dropout: float = 0.2
        self.attention_concat: bool = False
        self.attention_simple: bool = True
        self.mode: str = mode
        self.lr: float = 1e-3
        self.batch_size: int = dm.batch_size


class AspectClassifier(torch.nn.Module):

    def __init__(self, hparams: Union[HParams, AttributeDict], embeddings: Optional[torch.FloatTensor] = None) -> None:
        super().__init__()

        pprint(hparams)

        self.hparams = hparams

        # static embedding layer
        if embeddings is None:
            self.static_embedding = torch.nn.Embedding(hparams.vocab_size, hparams.embedding_dim)
        else:
            self.static_embedding = torch.nn.Embedding.from_pretrained(embeddings)

        input_dim = hparams.embedding_dim

        # pos embedding layer
        if hparams.pos_embedding:
            self.pos_embedding = torch.nn.Embedding(hparams.pos_vocab_size, hparams.pos_embedding_dim)
            input_dim += hparams.pos_embedding_dim

        # bert embedding layer
        if hparams.bert_embedding:
            self.bert_embedding = BertEmbedding(hparams.bert_model_name_or_path,
                                                finetune=hparams.bert_finetuning,
                                                layers_to_merge=hparams.bert_layers_to_merge,
                                                layer_pooling_strategy=hparams.bert_layer_pooling_strategy,
                                                wordpiece_pooling_strategy=hparams.bert_wordpiece_pooling_strategy)
            input_dim += self.bert_embedding.get_output_dim()

        # attention layer
        if self.hparams.attention:
            if self.hparams.attention_simple:
                self.attention = Attention(input_dim, hparams.attention_heads, hparams.attention_dropout)
            else:
                self.attention = TransformerEncoder(input_dim, hparams.attention_heads, hparams.attention_dropout)

        # recurrent layer
        self.lstm = torch.nn.LSTM(input_size=input_dim,
                                  hidden_size=hparams.hidden_dim,
                                  bidirectional=hparams.bidirectional,
                                  num_layers=hparams.num_layers,
                                  dropout=hparams.dropout if hparams.num_layers > 1 else 0,
                                  batch_first=True)

        lstm_output_dim = hparams.hidden_dim if not hparams.bidirectional else hparams.hidden_dim * 2
        if self.hparams.mode == "cd" and hparams.bidirectional:
            # in mode "cd" we concat the first and the last vector to summarize the bidirectional output
            lstm_output_dim *= 2

        # classification head
        classification_head_input_dim = lstm_output_dim

        if hparams.attention and hparams.attention_concat:
            classification_head_input_dim += input_dim

        self.classifier = torch.nn.Linear(classification_head_input_dim, hparams.num_classes)

        # regularization
        self.dropout = torch.nn.Dropout(hparams.dropout)

    def forward(self, batch) -> torch.Tensor:
        token_idxs = batch["token_idxs"]

        embeddings = self.static_embedding(token_idxs)

        if self.hparams.pos_embedding:
            pos_embeddings = self.pos_embedding(batch["pos_tag_idxs"])
            embeddings = torch.cat((embeddings, pos_embeddings), dim=-1)

        if self.hparams.bert_embedding:
            bert_embeddings = self.bert_embedding(batch)
            embeddings = torch.cat((embeddings, bert_embeddings), dim=-1)

        embeddings = self.dropout(embeddings)

        if self.hparams.attention:
            # boolean padding mask, w/ True in place of the pad token indexes, False otherwise
            key_padding_mask = ~(token_idxs != PAD_INDEX)
            attn_out = self.attention(embeddings, key_padding_mask)

        # either give the same input of the attention layer to the lstm as well,
        # or directly give the attention output as input of the lstm
        lstm_input_is_attn_out = self.hparams.attention and not self.hparams.attention_concat
        lstm_input = attn_out if lstm_input_is_attn_out else embeddings

        # lengths of the tokenized sentences w/o pad tokens
        lengths = torch.tensor([len(x) for x in token_idxs])

        if self.hparams.pack_lstm_input:
            # flatten the embeddings and packs them into a single sequence without padding
            # in order to reduce the lstm layer computing time and improve performance
            # see also: https://stackoverflow.com/a/56211056
            embeddings_packed = pack_padded_sequence(lstm_input,
                                                     lengths,
                                                     batch_first=True,
                                                     enforce_sorted=False)
            lstm_out_packed, _ = self.lstm(embeddings_packed)

            # unpack back the lstm's output as a normal padded batch
            lstm_out, _ = pad_packed_sequence(lstm_out_packed, batch_first=True)
        else:
            lstm_out, _ = self.lstm(embeddings)

        lstm_out = self.__get_summary_vectors(lstm_out, lengths)
        lstm_out = self.dropout(lstm_out)

        # the classification head takes as input either the lstm output
        # or its concatenation w/ the attention output
        if not self.hparams.attention or lstm_input_is_attn_out:
            classifier_input = lstm_out
        else:
            classifier_input = torch.cat([lstm_out, attn_out], dim=-1)

        out = self.classifier(classifier_input)

        return out

    def __get_summary_vectors(self, recurrent_out: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Returns the summary vectors of a recurrent layer output, based on the model mode (e.g. "ab" or "cd").
        For instance, in sequence labelling modes (e.g. "ab") all the vectors are returned, while for multi-class/labels
        classification only one vector is returned.

        Args:
            recurrent_out: output of a recurrent layer
            lengths: list of sequence lengths of each batch element without padding
        """
        if self.hparams.mode == "ab":
            return recurrent_out
        elif self.hparams.mode == "cd":
            batch_size, seq_len, hidden_size = recurrent_out.shape

            # flattening the recurrent output to have a long sequence of (batch_size x seq_len) vectors
            flattened_out = recurrent_out.reshape(-1, hidden_size)

            # tensor of the start offsets of each element in the batch
            sequences_offsets = torch.arange(batch_size) * seq_len

            # computing a tensor of the indices of the token in the last positions of each batch element
            last_vectors_indices = sequences_offsets + (lengths - 1)

            # summary vectors that summarize the elements in the batch
            summary_vectors = flattened_out[last_vectors_indices]

            if self.hparams.bidirectional:
                # concat the vectors from the token in the first positions of each batch element to the summary vectors
                summary_vectors = torch.cat((flattened_out[sequences_offsets], summary_vectors), dim=-1)

            return summary_vectors
