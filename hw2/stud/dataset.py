import pytorch_lightning as pl
import torch

from evaluate import read_dataset
from stud.constants import PAD_INDEX, UNK_TOKEN, PAD_TOKEN, UNK_INDEX
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import vocab, Vocab
from typing import *


def build_vocab(dataset: "ABSADataset", min_freq: int = 1) -> Vocab:
    counter = Counter()

    for sample in dataset:
        for token in sample["tokens"]:
            counter[token] += 1

    # min_freq is the minimum number of times that a token must appear in order to be part of the vocabulary
    vocabulary = vocab(counter, min_freq=min_freq)

    # add special tokens to handle padding and unknown words at testing time
    vocabulary.insert_token(UNK_TOKEN, UNK_INDEX)
    vocabulary.set_default_index(UNK_INDEX)

    vocabulary.insert_token(PAD_TOKEN, PAD_INDEX)

    return vocabulary


def build_label_vocab(dataset: "ABSADataset") -> Tuple[Vocab, Vocab, Vocab]:
    tag_counter = Counter()
    bio_counter = Counter()
    sentiment_counter = Counter()

    for sample in dataset:
        for i in range(len(sample["tokens"])):
            token = sample["tokens"][i]
            tag = sample["tags"][i]
            if token != PAD_TOKEN:
                tag_counter[tag] += 1
                bio_counter[tag[0]] += 1
                sentiment_counter[tag[2:]] += 1

    tag_vocabulary = vocab(tag_counter)
    bio_vocabulary = vocab(bio_counter)
    sentiment_vocabulary = vocab(sentiment_counter)

    tag_vocabulary.insert_token(PAD_TOKEN, PAD_INDEX)
    bio_vocabulary.insert_token(PAD_TOKEN, PAD_INDEX)
    sentiment_vocabulary.insert_token(PAD_TOKEN, PAD_INDEX)

    return tag_vocabulary, bio_vocabulary, sentiment_vocabulary


def padding_collate_fn(batch):
    token_idxs = [sample["token_idxs"] for sample in batch]
    bio_idxs = [sample["bio_idxs"] for sample in batch]
    sentiment_idxs = [sample["sentiment_idxs"] for sample in batch]
    tag_idxs = [sample["tag_idxs"] for sample in batch]

    padded_token_idxs = pad_sequence(token_idxs, batch_first=True, padding_value=PAD_INDEX)
    padded_bio_idxs = pad_sequence(bio_idxs, batch_first=True, padding_value=PAD_INDEX)
    padded_sentiment_idxs = pad_sequence(sentiment_idxs, batch_first=True, padding_value=PAD_INDEX)
    padded_tag_idxs = pad_sequence(tag_idxs, batch_first=True, padding_value=PAD_INDEX)

    return {
        "targets": [sample["targets"] for sample in batch],
        "text": [sample["text"] for sample in batch],
        "tokens": [sample["tokens"] for sample in batch],
        "tags": [sample["tags"] for sample in batch],
        "token_idxs": padded_token_idxs,
        "bio_idxs": padded_bio_idxs,
        "sentiment_idxs": padded_sentiment_idxs,
        "tag_idxs": padded_tag_idxs
    }


class ABSADataset(Dataset):

    def __init__(
            self,
            samples: List[Dict],
            tokenizer,
            vocabularies: Dict[str, Vocab] = None,
            isolate_targets: bool = False
    ) -> None:

        super().__init__()

        if isolate_targets:
            samples = self.__isolate_targets(samples)
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
            tokenizer,
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

    def __isolate_targets(self, samples: List[Dict]) -> List[Dict]:
        """
        Applies an augmentation on the samples list, duplicating the ones with
        more than one target and keeping only one of them per sample
        """

        augmented_samples = list()

        for sample in samples:
            if len(sample["targets"]) > 1:
                for target in sample["targets"]:
                    augmented_samples.append({"targets": [target], "text": sample["text"]})
            else:
                    augmented_samples.append(x)

        return augmented_samples

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


class ABSADataModule(pl.LightningDataModule):

    def __init__(self,
                 train_samples: List[Dict],
                 val_samples: List[Dict],
                 tokenizer,
                 vocabularies: Dict[str, Vocab] = None,
                 batch_size: int = 32,
                 augment_train: bool = True) -> None:
        super().__init__()
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.tokenizer = tokenizer
        self.vocabs = vocabularies
        self.batch_size = batch_size
        self.augment_train = augment_train

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_set = ABSADataset(self.train_samples,
                                     tokenizer=self.tokenizer,
                                     vocabularies=self.vocabs,
                                     isolate_targets=self.augment_train)

        self.val_set = ABSADataset(self.val_samples,
                                   tokenizer=self.tokenizer,
                                   vocabularies=self.vocabs)

        if self.vocabs is None:
            self.vocabs = dict()
            self.vocabs["text"] = build_vocab(self.train_set + self.val_set)
            self.vocabs["tag"], self.vocabs["bio"], self.vocabs["sentiment"] = build_label_vocab(self.train_set)

        self.train_set.encode_samples(self.vocabs)
        self.val_set.encode_samples(self.vocabs)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set,
                          shuffle=True,
                          batch_size=self.batch_size,
                          collate_fn=padding_collate_fn)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_set,
                          shuffle=False,
                          batch_size=self.batch_size,
                          collate_fn=padding_collate_fn)
