from typing import *

import torch
from torch.utils.data import Dataset
from torchtext.vocab import Vocab
from transformers import PreTrainedTokenizer

from evaluate import read_dataset


class ABSADataset(Dataset):

    def __init__(
            self,
            samples: List[Dict],
            tokenizer: PreTrainedTokenizer,
            vocabularies: Dict[str, Vocab] = None
    ) -> None:

        super().__init__()

        self.samples = {key: [dic[key] for dic in samples] for key in samples[0]}
        self.tokenizer = tokenizer
        self.encoded_samples = {
            "tokens": list(),
            "tags": list(),
            "token_idxs": list(),
            "bio_idxs": list(),
            "sentiment_idxs": list(),
            "tag_idxs": list()
        }

        self.__preprocess_samples()

        if vocabularies is not None:
            self.encode_samples(vocabularies)

    @classmethod
    def from_file(
            cls,
            path: str,
            tokenizer: PreTrainedTokenizer,
            vocabularies: Dict[str, Vocab] = None
    ) -> "ABSADataset":

        return cls(read_dataset(path), tokenizer, vocabularies)

    def encode_samples(self, vocabularies: Dict[str, Vocab]) -> None:

        token_idxs = list()
        bio_idxs = list()
        tag_idxs = list()
        sentiment_idxs = list()

        for tokens, tags in zip(self.encoded_samples["tokens"], self.encoded_samples["tags"]):

            sample_token_idxs = list()
            sample_bio_idxs = list()
            sample_sentiment_idxs = list()
            sample_tag_idxs = list()

            for token, tag in zip(tokens, tags):
                sample_token_idxs.append(vocabularies["text"][token])
                sample_tag_idxs.append(vocabularies["tag"][tag])
                sample_bio_idxs.append(vocabularies["bio"][tag[0]])
                sample_sentiment_idxs.append(vocabularies["sentiment"][tag[2:]])

            token_idxs.append(torch.tensor(sample_token_idxs))
            bio_idxs.append(torch.tensor(sample_bio_idxs))
            sentiment_idxs.append(torch.tensor(sample_sentiment_idxs))
            tag_idxs.append(torch.tensor(sample_tag_idxs))

        self.encoded_samples["token_idxs"] = token_idxs
        self.encoded_samples["bio_idxs"] = bio_idxs
        self.encoded_samples["sentiment_idxs"] = sentiment_idxs
        self.encoded_samples["tag_idxs"] = tag_idxs

    @staticmethod
    def decode_output(predictions: torch.Tensor, label_vocabulary: Vocab):

        decoded_predictions = list()

        for indices in predictions:
            # vocabulary integer to string used to obtain the corresponding label from the index
            decoded_predictions.append([label_vocabulary.get_itos()[i] for i in indices])

        return decoded_predictions

    def __preprocess_samples(self) -> None:

        tokens = list()
        tags = list()

        for sample_text, sample_targets in zip(self.samples["text"], self.samples["targets"]):
            sample_tokens = self.tokenizer.tokenize(sample_text)
            sample_tags = self.__tag_sample(sample_text, sample_targets, sample_tokens)
            tokens.append(sample_tokens)
            tags.append(sample_tags)

        self.encoded_samples["tokens"] = tokens
        self.encoded_samples["tags"] = tags

    def __tag_sample(self, sample_text: str, sample_targets: List, sample_tokens: List[str]) -> List[str]:

        tags = ["O" for token in range(len(sample_tokens))]

        for (start, end), target, tag in sample_targets:
            target_subwords_tokens = self.tokenizer.tokenize(sample_text[start:end])
            target_previous_tokens = self.tokenizer.tokenize(sample_text[:start])
            n_prev_tokens = len(target_previous_tokens)

            target_positions = list(range(n_prev_tokens, n_prev_tokens + len(target_subwords_tokens)))

            for i, t_pos in enumerate(target_positions):
                tags[t_pos] = f"B-{tag}" if i == 0 else f"I-{tag}"

        return tags

    def __len__(self) -> int:
        return len(self.samples["text"])

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, List[str], str]]:

        token_idxs = self.encoded_samples["token_idxs"]
        bio_idxs = self.encoded_samples["bio_idxs"]
        sentiment_idxs = self.encoded_samples["sentiment_idxs"]
        tag_idxs = self.encoded_samples["tag_idxs"]

        return {
            "targets": self.samples["targets"][idx],
            "text": self.samples["text"][idx],
            "tokens": self.encoded_samples["tokens"][idx],
            "tags": self.encoded_samples["tags"][idx],
            "token_idxs": token_idxs[idx] if len(token_idxs) != 0 else None,
            "bio_idxs": bio_idxs[idx] if len(bio_idxs) != 0 else None,
            "sentiment_idxs": sentiment_idxs[idx] if len(sentiment_idxs) != 0 else None,
            "tag_idxs": tag_idxs[idx] if len(tag_idxs) != 0 else None
        }
