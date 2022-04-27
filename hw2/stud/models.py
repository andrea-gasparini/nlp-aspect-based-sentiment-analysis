from pprint import pprint
from typing import *

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from transformers import BertModel, AutoTokenizer

from stud import utils
from stud.constants import PAD_INDEX


class BertEmbedding(torch.nn.Module):
    BERT_OUT_DIM = 768

    def __init__(self,
                 pretrained_model_name_or_path: str = "bert-base-cased",
                 layer_pooling_strategy: str = "last",
                 layers_to_merge: Sequence[int] = (-1, -2, -3, -4),
                 wordpiece_pooling_strategy: str = "mean") -> None:
        super().__init__()

        self.layers_to_merge = layers_to_merge
        self.layer_pooling_strategy = layer_pooling_strategy
        self.wordpiece_pooling_strategy = wordpiece_pooling_strategy
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.model = BertModel.from_pretrained(pretrained_model_name_or_path, return_dict=True)
        self.model.eval()

    def forward(self, batch: Dict[str, Union[Tensor, List]]) -> Tensor:

        encoding = self.tokenizer(batch["tokens"],
                                  return_tensors="pt",
                                  padding=True,
                                  truncation=False,
                                  is_split_into_words=True)

        batch_device = utils.get_device(batch["token_idxs"])

        input_ids = encoding["input_ids"].to(batch_device)

        self.model.to(batch_device)
        bert_out = self.model(input_ids, output_hidden_states=True)
        bert_out = self.merge_hidden_states(bert_out.hidden_states, self.layers_to_merge)

        word_ids = [sample.word_ids for sample in encoding.encodings]

        aggregated_bert_out = [self.aggregate_wordpiece_vectors(word_ids, bert_out) for word_ids, bert_out in
                               zip(word_ids, bert_out)]

        bert_out = [self.merge_wordpiece_vectors(pairs) for pairs in aggregated_bert_out]
        bert_out = pad_sequence([torch.stack(tensor) for tensor in bert_out], batch_first=True, padding_value=PAD_INDEX)

        # filter special tokens from the word ids list ...
        filtered_word_ids = [[w_id for w_id in w_ids if w_id is not None] for w_ids in word_ids]
        # ... and consecutive equal ids, i.e. WordPieces of the same word
        for w_ids in filtered_word_ids:
            for i in range(len(w_ids) - 1, -1, -1):
                if w_ids[i] == w_ids[i - 1]:
                    w_ids.pop(i)

        lengths = [len(x) for x in filtered_word_ids]

        encoding_mask = list()

        for w_ids in filtered_word_ids:
            tokens = len(w_ids) * [True]
            # add False for both [CLS] and [SEP]
            bert_tokens = [False] + tokens + [False]
            # add False as [PAD] to match the padded batch len
            padded_tokens = bert_tokens + [False] * (len(bert_out[0]) - 2 - len(tokens))
            encoding_mask.append(torch.tensor(padded_tokens))

        encoding_mask = torch.stack(encoding_mask)

        return self.remove_bert_tokens(bert_out, encoding_mask, lengths)

    def get_output_dim(self):
        """
        Returns the final embedding dimension based on the layer pooling strategy
        """
        if self.layer_pooling_strategy == "concat":
            return self.BERT_OUT_DIM * len(self.layers_to_merge)
        else:
            return self.BERT_OUT_DIM

    def aggregate_wordpiece_vectors(self, word_ids: List[int], vectors: Tensor) -> List[List[Tuple[int, Tensor]]]:
        """
        Aggregate subwords WordPiece vectors (which are identified by consecutive equal word_ids)

        e.g. word_ids = [0, 1, 1, 2] --> [[(0, tensor([...]))],
                                          [(1, tensor([...])),
                                           (1, tensor([...]))],
                                          [(2, tensor([...]))]]
        """
        aggregated_tokens = list()
        token = [(word_ids[0], vectors[0])]

        for w_id, vector in zip(word_ids[1:], vectors[1:]):
            vector = vector
            if w_id is not None and w_id == token[-1][0]:
                token.append((w_id, vector))
            else:
                aggregated_tokens.append(token)
                token = [(w_id, vector)]

        if len(token) > 0:
            aggregated_tokens.append(token)

        return aggregated_tokens

    def merge_hidden_states(self, layers: Tuple[Tensor], to_merge: Sequence[int]) -> Tensor:
        """
        Applies the pooling strategy to the hidden states

        Args:
            layers: hidden states outputted from the transformer model
            to_merge: indexes of which hidden layers to apply the pooling strategy on
        """
        if self.layer_pooling_strategy == "last":
            return layers[-1]
        elif self.layer_pooling_strategy == "concat":
            return torch.cat([layers[l] for l in to_merge], dim=-1)
        elif self.layer_pooling_strategy == "sum":
            return torch.stack([layers[l] for l in to_merge], dim=0).sum(dim=0)
        elif self.layer_pooling_strategy == "mean":
            return torch.stack([layers[l] for l in to_merge], dim=0).mean(dim=0)
        else:
            raise NotImplementedError(f"Layers pooling strategy {self.layer_pooling_strategy} not implemented.")

    def merge_wordpiece_vectors(self, wordpiece_vector_pairs: List[List[Tuple[int, Tensor]]]) -> List[Tensor]:
        """
        Applies the pooling strategy to the aggregated subwords WordPiece vectors
        """
        vectors = list()
        for pairs in wordpiece_vector_pairs:
            _, pair_vectors = zip(*pairs)

            if self.wordpiece_pooling_strategy == "mean":
                vector = torch.stack(pair_vectors).mean(dim=0)
            elif self.wordpiece_pooling_strategy == "first+last":
                vector = torch.cat([pair_vectors[0], pair_vectors[-1]], dim=0)
            else:
                raise NotImplementedError(
                    f"Subword pooling strategy {self.wordpiece_pooling_strategy} not implemented.")

            vectors.append(vector)
        return vectors

    def remove_bert_tokens(self, encodings, encoding_mask, lengths) -> Tensor:
        """
        Remove [CLS] and [SEP] tokens
        """
        flattened_filtered_encodings = encodings[encoding_mask]
        encodings = flattened_filtered_encodings.split(lengths)
        return pad_sequence(encodings, batch_first=True, padding_value=PAD_INDEX)


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
                                                layer_pooling_strategy=hparams.bert_layer_pooling_strategy,
                                                wordpiece_pooling_strategy=hparams.bert_wordpiece_pooling_strategy)
            input_dim += self.bert_embedding.get_output_dim()

        # recurrent layer
        self.lstm = torch.nn.LSTM(input_size=input_dim,
                                  hidden_size=hparams.hidden_dim,
                                  bidirectional=hparams.bidirectional,
                                  num_layers=hparams.num_layers,
                                  dropout=hparams.dropout if hparams.num_layers > 1 else 0,
                                  batch_first=True)

        lstm_output_dim = hparams.hidden_dim if hparams.bidirectional is False else hparams.hidden_dim * 2

        # classification head
        self.classifier = torch.nn.Linear(lstm_output_dim, hparams.num_classes)

        # regularization
        self.dropout = torch.nn.Dropout(hparams.dropout)

    def forward(self, batch) -> torch.Tensor:
        token_idxs = batch["token_idxs"]

        embeddings = self.static_embedding(token_idxs)
        embeddings = self.dropout(embeddings)

        if self.hparams.pos_embedding:
            pos_embeddings = self.pos_embedding(batch["pos_idxs"])
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

        if self.hparams.pack_lstm_input:
            # flatten the embeddings and packs them into a single sequence without padding
            # in order to reduce the lstm layer computing time and improve performance
            # see also: https://stackoverflow.com/a/56211056
            lengths = torch.tensor([len(x) for x in token_idxs])
            embeddings_packed = pack_padded_sequence(embeddings,
                                                     lengths,
                                                     batch_first=True,
                                                     enforce_sorted=False)
            out_packed, _ = self.lstm(embeddings_packed)

            # unpack back the lstm's output as a normal padded batch
            out, _ = pad_packed_sequence(out_packed, batch_first=True)
        else:
            out, _ = self.lstm(embeddings)

        out = self.dropout(out)
        out = self.classifier(out)

        return out
