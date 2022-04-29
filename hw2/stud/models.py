from pprint import pprint
from typing import *

import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from stud.constants import PAD_INDEX
from stud.layers import BertEmbedding, Attention, TransformerEncoder


class AspectTermsClassifier(torch.nn.Module):

    def __init__(self, hparams, embeddings: Optional[torch.FloatTensor] = None) -> None:
        super().__init__()

        pprint(hparams)

        self.hparams = hparams

        # embedding layer
        if embeddings is None:
            self.static_embedding = torch.nn.Embedding(hparams.vocab_size, hparams.embedding_dim)
        else:
            self.static_embedding = torch.nn.Embedding.from_pretrained(embeddings)

        input_dim = hparams.embedding_dim

        if hparams.pos_embedding:
            self.pos_embedding = torch.nn.Embedding(hparams.pos_vocab_size, hparams.pos_embedding_dim)
            input_dim += hparams.pos_embedding_dim

        if hparams.bert_embedding:
            self.bert_embedding = BertEmbedding(hparams.bert_model_name_or_path,
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
        embeddings = self.dropout(embeddings)

        if self.hparams.pos_embedding:
            pos_embeddings = self.pos_embedding(batch["pos_tag_idxs"])
            pos_embeddings = self.dropout(pos_embeddings)
            embeddings = torch.cat((embeddings, pos_embeddings), dim=-1)

        if self.hparams.bert_embedding:
            if not self.hparams.bert_finetuning:
                with torch.no_grad():
                    bert_embeddings = self.bert_embedding(batch)
            else:
                bert_embeddings = self.bert_embedding(batch)
            bert_embeddings = self.dropout(bert_embeddings)

            embeddings = torch.cat((embeddings, bert_embeddings), dim=-1)

        if self.hparams.attention:
            key_padding_mask = ~(token_idxs != PAD_INDEX)
            attn_out = self.attention(embeddings, key_padding_mask)

        lstm_input_is_attn_out = self.hparams.attention and not self.hparams.attention_concat
        lstm_input = attn_out if lstm_input_is_attn_out else embeddings

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
        if self.hparams.mode in ["ab", "a", "b"]:
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
