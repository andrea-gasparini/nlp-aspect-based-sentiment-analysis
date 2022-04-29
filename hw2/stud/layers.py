from typing import *

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from transformers import BertModel, AutoTokenizer

from stud import utils
from stud.constants import PAD_INDEX


class TransformerEncoder(torch.nn.Module):
	"""
	A multi-head attention based encoder, implements a single layer of the Transformer encoder
	described in the paper: `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`
	"""
	FEED_FORWARD_INNER_DIM = 2048

	def __init__(self, embed_dim: int = 512, num_heads: int = 8, dropout: float = 0.1) -> None:
		super().__init__()

		self.self_attn = torch.nn.MultiheadAttention(embed_dim, num_heads, dropout, batch_first=True)

		self.feedforward = torch.nn.Sequential(torch.nn.Linear(embed_dim, self.FEED_FORWARD_INNER_DIM),
											   torch.nn.ReLU(),
											   torch.nn.Linear(self.FEED_FORWARD_INNER_DIM, embed_dim))

		self.norm1 = torch.nn.LayerNorm(embed_dim)
		self.norm2 = torch.nn.LayerNorm(embed_dim)

		self.dropout = torch.nn.Dropout(dropout)

	def forward(self, inputs: Tensor, key_padding_mask: Optional[Tensor] = None) -> torch.Tensor:
		
		out1, _ = self.self_attn(inputs, inputs, inputs, key_padding_mask=key_padding_mask)
		out1 = self.dropout(out1)
		out1 = self.norm1(inputs + out1)

		out2 = self.feedforward(out1)
		out2 = self.dropout(out2)
		out = self.norm2(out1 + out2)

		return out


class Attention(torch.nn.Module):
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()

        self.self_attn = torch.nn.MultiheadAttention(embed_dim, num_heads, dropout, batch_first=True)
        self.attention_dropout = torch.nn.Dropout(dropout)

    def forward(self, inputs: Tensor, key_padding_mask: Optional[Tensor] = None) -> torch.Tensor:
        
        out, _ = self.self_attn(inputs, inputs, inputs, key_padding_mask=key_padding_mask)
        out = self.attention_dropout(out)

        return out


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