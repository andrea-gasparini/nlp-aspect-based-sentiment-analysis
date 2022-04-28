from typing import *

import nltk
import pytorch_lightning as pl
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import vocab, Vocab

from evaluate import read_dataset
from stud.constants import PAD_INDEX, UNK_TOKEN, PAD_TOKEN


def build_vocab(dataset: "ABSADataset",
                min_freq: int = 1,
                unk_token: bool = True,
                get_tokens_fn: Callable[[Dict], List[str]] = lambda sample: sample["tokens"]) -> Vocab:
    """
    Creates a Vocab object from an ABSADataset

    Args:
        dataset: dataset to build the vocabulary from its samples
        min_freq: min number of times a token must appear in order to be included in the vocabulary
        unk_token: whether to insert or not a default "unknown" token
        get_tokens_fn: function to retrieve the list of tokens to insert in the vocabulary from a dataset sample
    """
    counter = Counter()

    for sample in dataset:
        for token in get_tokens_fn(sample):
            counter[token] += 1

    vocabulary = vocab(counter, min_freq=min_freq, specials=[UNK_TOKEN] if unk_token else None)
    vocabulary.insert_token(PAD_TOKEN, PAD_INDEX)

    if unk_token:
        vocabulary.set_default_index(vocabulary[UNK_TOKEN])

    return vocabulary


def build_pos_vocab(dataset: "ABSADataset") -> Vocab:
    return build_vocab(dataset, unk_token=False, get_tokens_fn=lambda sample: sample["pos_tags"])


def build_label_vocabs(dataset: "ABSADataset") -> Tuple[Vocab, Vocab, Vocab]:
    return build_vocab(dataset, unk_token=False, get_tokens_fn=lambda sample: sample["tags"]), \
           build_vocab(dataset, unk_token=False, get_tokens_fn=lambda sample: [tag[0] for tag in sample["tags"]]), \
           build_vocab(dataset, unk_token=False, get_tokens_fn=lambda sample: [tag[2:] for tag in sample["tags"]])


def padding_collate_fn(batch):
    token_idxs = [sample["token_idxs"] for sample in batch]
    pos_tag_idxs = [sample["pos_tag_idxs"] for sample in batch]
    bio_idxs = [sample["bio_idxs"] for sample in batch]
    sentiment_idxs = [sample["sentiment_idxs"] for sample in batch]
    tag_idxs = [sample["tag_idxs"] for sample in batch]

    padded_token_idxs = pad_sequence(token_idxs, batch_first=True, padding_value=PAD_INDEX)
    padded_pos_tag_idxs = pad_sequence(pos_tag_idxs, batch_first=True, padding_value=PAD_INDEX)
    padded_bio_idxs = pad_sequence(bio_idxs, batch_first=True, padding_value=PAD_INDEX)
    padded_sentiment_idxs = pad_sequence(sentiment_idxs, batch_first=True, padding_value=PAD_INDEX)
    padded_tag_idxs = pad_sequence(tag_idxs, batch_first=True, padding_value=PAD_INDEX)

    return {
        "targets": [sample["targets"] for sample in batch],
        "text": [sample["text"] for sample in batch],
        "tokens": [sample["tokens"] for sample in batch],
        "tags": [sample["tags"] for sample in batch],
        "token_idxs": padded_token_idxs,
        "pos_tag_idxs": padded_pos_tag_idxs,
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

        self.raw_samples = samples
        self.tokenizer = tokenizer
        self.encoded_samples = {
            "tokens": list(),
            "pos_tags": list(),
            "tags": list(),
            "pos_tag_idxs": list(),
            "token_idxs": list(),
            "bio_idxs": list(),
            "sentiment_idxs": list(),
            "tag_idxs": list()
        }

        self.__preprocess_samples(isolate_targets)

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
        pos_tag_idxs = list()
        bio_idxs = list()
        tag_idxs = list()
        sentiment_idxs = list()

        for tokens, pos_tags, tags in zip(self.encoded_samples["tokens"],
                                          self.encoded_samples["pos_tags"],
                                          self.encoded_samples["tags"]):

            sample_token_idxs = list()
            sample_pos_tag_idxs = list()
            sample_bio_idxs = list()
            sample_sentiment_idxs = list()
            sample_tag_idxs = list()

            for token, pos_tag, tag in zip(tokens, pos_tags, tags):
                sample_token_idxs.append(vocabularies["text"][token])
                sample_pos_tag_idxs.append(vocabularies["pos"][pos_tag])
                sample_tag_idxs.append(vocabularies["tag"][tag])
                sample_bio_idxs.append(vocabularies["bio"][tag[0]])
                sample_sentiment_idxs.append(vocabularies["sentiment"][tag[2:]])

            token_idxs.append(torch.tensor(sample_token_idxs))
            pos_tag_idxs.append(torch.tensor(sample_pos_tag_idxs))
            bio_idxs.append(torch.tensor(sample_bio_idxs))
            sentiment_idxs.append(torch.tensor(sample_sentiment_idxs))
            tag_idxs.append(torch.tensor(sample_tag_idxs))

        self.encoded_samples["token_idxs"] = token_idxs
        self.encoded_samples["pos_tag_idxs"] = pos_tag_idxs
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
                augmented_samples.append(sample)

        return augmented_samples

    def __preprocess_samples(self, isolate_targets: bool = False) -> None:

        samples = self.raw_samples
        tokens = list()
        pos_tags = list()
        tags = list()

        if isolate_targets:
            samples = self.__isolate_targets(self.raw_samples)

        self.samples = {key: [dic[key] for dic in samples] for key in samples[0]}

        for sample_text, sample_targets in zip(self.samples["text"], self.samples["targets"]):
            sample_tokens = self.tokenizer.tokenize(sample_text)
            sample_pos_tags = [pos[1] for pos in nltk.pos_tag(sample_tokens)]
            sample_tags = self.__tag_sample(sample_text, sample_targets, sample_tokens)
            tokens.append(sample_tokens)
            pos_tags.append(sample_pos_tags)
            tags.append(sample_tags)

        self.encoded_samples["pos_tags"] = pos_tags
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

        return {
            "targets": self.samples["targets"][idx],
            "text": self.samples["text"][idx],
            "tokens": self.encoded_samples["tokens"][idx],
            "pos_tags": self.encoded_samples["pos_tags"][idx],
            "tags": self.encoded_samples["tags"][idx],
            "token_idxs": self.__get_indices("token_idxs", idx),
            "pos_tag_idxs": self.__get_indices("pos_tag_idxs", idx),
            "bio_idxs": self.__get_indices("bio_idxs", idx),
            "sentiment_idxs": self.__get_indices("sentiment_idxs", idx),
            "tag_idxs": self.__get_indices("tag_idxs", idx)
        }

    def __get_indices(self, key: str, index: int) -> Optional[torch.Tensor]:
        return self.encoded_samples[key][index] if len(self.encoded_samples[key]) != 0 else None


class ABSADataModule(pl.LightningDataModule):

    def __init__(self,
                 train_samples: List[Dict],
                 val_samples: List[Dict],
                 tokenizer,
                 vocabularies: Dict[str, Vocab] = None,
                 batch_size: int = 32,
                 augment_train: bool = True,
                 num_workers: int = 0,
                 pin_memory: bool = False) -> None:
        super().__init__()
        self.train_set = None
        self.val_set = None
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.tokenizer = tokenizer
        self.vocabs = vocabularies
        self.batch_size = batch_size
        self.augment_train = augment_train
        self.num_workers = num_workers
        self.pin_memory = pin_memory

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
            self.vocabs["pos"] = build_pos_vocab(self.train_set + self.val_set)
            self.vocabs["tag"], self.vocabs["bio"], self.vocabs["sentiment"] = build_label_vocabs(self.train_set)

        self.train_set.encode_samples(self.vocabs)
        self.val_set.encode_samples(self.vocabs)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set,
                          shuffle=True,
                          batch_size=self.batch_size,
                          collate_fn=padding_collate_fn,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_set,
                          shuffle=False,
                          batch_size=self.batch_size,
                          collate_fn=padding_collate_fn,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory)
