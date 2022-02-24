from pprint import pprint
from typing import *

import pytorch_lightning as pl
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import BertModel, AutoTokenizer

from stud.constants import PAD_INDEX


class BertEmbedding(pl.LightningModule):

    def __init__(self, pretrained_model_name_or_path: str = "bert-base-cased") -> None:
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.model = BertModel.from_pretrained(pretrained_model_name_or_path, return_dict=True)
        self.model.to(self.device)
        self.model.eval()

    def forward(self, batch: Dict[str, Union[torch.Tensor, List]]) -> torch.Tensor:

        encoding = self.tokenizer(batch["text"],
                                  return_tensors="pt",
                                  padding=True,
                                  truncation=False,
                                  is_split_into_words=False)

        word_ids = [sample.word_ids for sample in encoding.encodings]

        bert_out = self.model(encoding["input_ids"].to(self.device), output_hidden_states=True)["last_hidden_state"]

        aggregated_bert_out = [self.aggregate_wordpiece_vectors(word_ids, bert_out) for word_ids, bert_out in
                               zip(word_ids, bert_out)]

        bert_out = [self.merge_wordpiece_vectors(pairs) for pairs in aggregated_bert_out]
        bert_out = pad_sequence([torch.stack(tensor) for tensor in bert_out], batch_first=True, padding_value=PAD_INDEX)

        lengths = [len(x) for x in batch["tokens"]]

        encoding_mask = list()

        for w_ids in batch["tokens"]:
            tokens = len(w_ids) * [True]
            # add False for both [CLS] and [SEP]
            bert_tokens = [False] + tokens + [False]
            # add False as [PAD] to match the padded batch len
            padded_tokens = bert_tokens + [False] * (len(batch["token_idxs"][0]) - len(tokens))
            encoding_mask.append(torch.tensor(padded_tokens))

        encoding_mask = torch.stack(encoding_mask)

        return self.remove_bert_tokens(bert_out, encoding_mask, lengths)

    def aggregate_wordpiece_vectors(
            self,
            word_ids: List[int],
            vectors: torch.Tensor
    ) -> List[List[Tuple[int, torch.Tensor]]]:
        """
        Aggregate subwords WordPiece vectors (which are identified by consecutives equal word_ids)

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

    def merge_wordpiece_vectors(
            self,
            wordpiece_vector_pairs: List[List[Tuple[int, torch.Tensor]]]
    ) -> List[torch.Tensor]:
        """
        Merge, by arithmetic mean, the aggregated subwords WordPiece vectors
        given by the `aggregate_wordpiece_vectors` function
        """
        vectors = list()
        for pairs in wordpiece_vector_pairs:
            pair_subwords, pair_vectors = zip(*pairs)
            # arithmetic mean of the sub-words embeddings # TODO try weighted average
            vector = torch.stack(pair_vectors).mean(dim=0)
            for _ in range(len(pair_vectors)):
                vectors.append(vector)
        return vectors

    def remove_bert_tokens(self, encodings, encoding_mask, lengths) -> torch.Tensor:
        """
        Remove [CLS] and [SEP] tokens
        """
        flattened_filtered_encodings = encodings[encoding_mask]
        encodings = flattened_filtered_encodings.split(lengths)
        return pad_sequence(encodings, batch_first=True, padding_value=PAD_INDEX)


class AspectTermsClassifier(torch.nn.Module):

    def __init__(self, hparams, embeddings: Optional[torch.FloatTensor] = None):
        super().__init__()

        pprint(hparams)

        self.hparams = hparams

        # embedding layer
        if hparams.bert_embedding:
            self.word_embedding = BertEmbedding(hparams.bert_model_name_or_path)
        elif embeddings is None:
            self.word_embedding = torch.nn.Embedding(hparams.vocab_size, hparams.embedding_dim)
        else:
            self.word_embedding = torch.nn.Embedding.from_pretrained(embeddings)

        # recurrent layer
        self.lstm = torch.nn.LSTM(hparams.embedding_dim,
                                  hparams.hidden_dim,
                                  bidirectional=hparams.bidirectional,
                                  num_layers=hparams.num_layers,
                                  dropout=hparams.dropout if hparams.num_layers > 1 else 0)

        lstm_output_dim = hparams.hidden_dim if hparams.bidirectional is False else hparams.hidden_dim * 2

        # classification head
        self.classifier = torch.nn.Linear(lstm_output_dim, hparams.num_classes)

        # regularization
        self.dropout = torch.nn.Dropout(hparams.dropout)
        self.relu = torch.nn.ReLU()

    def forward(self, batch):
        token_idxs = batch["token_idxs"]

        if self.hparams.bert_embedding:
            with torch.no_grad():
                embeddings = self.word_embedding(batch)
        else:
            embeddings = self.word_embedding(token_idxs)

        embeddings = self.dropout(embeddings)
        out, (h, c) = self.lstm(embeddings)
        out = self.dropout(out)
        out = self.classifier(out)
        return out
