from typing import *

import nltk
import pytorch_lightning as pl
import torch
from torch.nn.utils import rnn
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import vocab, Vocab

from evaluate import read_dataset
from stud.constants import PAD_INDEX, UNK_TOKEN, PAD_TOKEN


def build_vocab(dataset: "ABSADataset",
                min_freq: int = 1,
                unk_token: bool = True,
                pad_token: bool = True,
                get_tokens_fn: Callable[[Dict], List[str]] = lambda sample: sample["tokens"]) -> Vocab:
    """
    Creates a Vocab object from an ABSADataset

    Args:
        dataset: dataset to build the vocabulary from its samples
        min_freq: min number of times a token must appear in order to be included in the vocabulary
        unk_token: whether to insert or not a default "unknown" token
        pad_token: whether to insert or not a default "padding" token
        get_tokens_fn: function to retrieve the list of tokens to insert in the vocabulary from a dataset sample
    """
    counter = Counter()

    for sample in dataset:
        for token in get_tokens_fn(sample):
            counter[token] += 1

    vocabulary = vocab(counter, min_freq=min_freq, specials=[UNK_TOKEN] if unk_token else None)

    if pad_token:
        vocabulary.insert_token(PAD_TOKEN, PAD_INDEX)

    if unk_token:
        vocabulary.set_default_index(vocabulary[UNK_TOKEN])

    return vocabulary


def build_pos_vocab(dataset: "ABSADataset") -> Vocab:
    return build_vocab(dataset, unk_token=False, get_tokens_fn=lambda sample: sample["pos_tags"])


def build_label_vocabs(dataset: "ABSADataset") -> Vocab:
    return build_vocab(dataset, unk_token=False, get_tokens_fn=lambda sample: sample["tags"])


def build_category_vocabs(dataset: "ABSADataset") -> Tuple[Vocab, Vocab]:
    return build_vocab(dataset, unk_token=False, pad_token=False,
                       get_tokens_fn=lambda sample: [category[0] for category in sample["categories"]]), \
           build_vocab(dataset, unk_token=False, pad_token=False,
                       get_tokens_fn=lambda sample: [category[1] for category in sample["categories"]])


def get_from_batch(batch, key: str = "token_idxs") -> Union[torch.Tensor, List[str], str]:
    return [sample[key] for sample in batch]


def pad_sequence(sequences: Union[List[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    return rnn.pad_sequence(sequences, batch_first=True, padding_value=PAD_INDEX)


def padding_collate_fn(batch):

    category_idxs = get_from_batch(batch, "category_idxs")
    tag_idxs = get_from_batch(batch, "tag_idxs")

    return {
        "targets": get_from_batch(batch, "targets"),
        "text": get_from_batch(batch, "text"),
        "tokens": get_from_batch(batch, "tokens"),
        "tags": get_from_batch(batch, "tags"),
        "categories": get_from_batch(batch, "categories"),
        "token_idxs": pad_sequence(get_from_batch(batch, "token_idxs")),
        "pos_tag_idxs": pad_sequence(get_from_batch(batch, "pos_tag_idxs")),
        "tag_idxs": pad_sequence(tag_idxs) if tag_idxs.count(None) != len(tag_idxs) else None,
        "category_idxs": torch.stack(category_idxs) if category_idxs.count(None) != len(category_idxs) else None
    }


class ABSADataset(Dataset):

    def __init__(
            self,
            samples: List[Dict],
            tokenizer,
            vocabularies: Dict[str, Vocab] = None,
            has_categories: bool = False,
            no_labels: bool = False
    ) -> None:

        super().__init__()

        self.raw_samples = samples
        self.tokenizer = tokenizer
        self.has_categories = has_categories
        self.no_labels = no_labels
        self.encoded_samples = {
            "tokens": list(),
            "pos_tags": list(),
            "tags": list(),
            "categories": list(),
            "pos_tag_idxs": list(),
            "token_idxs": list(),
            "tag_idxs": list(),
            "category_idxs": list()
        }

        self.__preprocess_samples()

        if vocabularies is not None:
            self.encode_samples(vocabularies)

    @classmethod
    def from_file(
            cls,
            path: str,
            tokenizer,
            vocabularies: Dict[str, Vocab] = None,
            has_categories: bool = False,
            no_labels: bool = False
    ) -> "ABSADataset":

        return cls(read_dataset(path), tokenizer, vocabularies, has_categories, no_labels)

    def encode_samples(self, vocabularies: Dict[str, Vocab]) -> None:
        """
        Encodes tokens, pos tags and labels as numeric indices through the mapping defined in the given vocabularies.
        """

        token_idxs, pos_tag_idxs, tag_idxs, category_idxs  = list(), list(), list(), list()

        for i in range(len(self.raw_samples)):

            sample_token_idxs, sample_pos_tag_idxs, sample_tag_idxs = list(), list(), list()

            for token, pos_tag, tag in zip(self.encoded_samples["tokens"][i],
                                           self.encoded_samples["pos_tags"][i],
                                           self.encoded_samples["tags"][i]):
                sample_token_idxs.append(vocabularies["text"][token])
                sample_pos_tag_idxs.append(vocabularies["pos"][pos_tag])

                if not self.no_labels:
                    sample_tag_idxs.append(vocabularies["tag"][tag])

            token_idxs.append(torch.tensor(sample_token_idxs))
            pos_tag_idxs.append(torch.tensor(sample_pos_tag_idxs))

            if not self.no_labels:
                tag_idxs.append(torch.tensor(sample_tag_idxs))

                if self.has_categories:
                    # build a `n_categories` x `n_polarities+1` (5x5) tensor,
                    # where the first dimension corresponds to the categories and the second to their polarities.
                    # A value of 1 at indices `[i, j]` means the sample has category of index `i` with polarity of index `j`
                    categories_vocab = vocabularies["categories"]
                    polarities_vocab = vocabularies["category_polarities"]
                    sample_category_idxs = torch.zeros(len(categories_vocab), len(polarities_vocab) + 1)
                    # A value of 1 at indices `[i, -1]` means the sample do not have a category of index `i`
                    sample_category_idxs[:, -1] = 1

                    for category, polarity in self.encoded_samples["categories"][i]:
                        sample_category_idxs[categories_vocab[category], -1] = 0
                        sample_category_idxs[categories_vocab[category], polarities_vocab[polarity]] = 1

                    category_idxs.append(sample_category_idxs)

        self.encoded_samples["token_idxs"] = token_idxs
        self.encoded_samples["pos_tag_idxs"] = pos_tag_idxs

        if not self.no_labels:
            self.encoded_samples["tag_idxs"] = tag_idxs

            if self.has_categories:
                self.encoded_samples["category_idxs"] = category_idxs

    @staticmethod
    def decode_output(predictions: torch.Tensor,
                      label_vocabulary: Vocab,
                      polarity_vocabulary: Optional[Vocab] = None,
                      mode: str = "ab") -> Union[List[List[Tuple[str, str]]], List[List[str]]]:
        """
        Decodes the labels/predictions from their indices format back to the original string form,
        through the mapping defined in the given vocabularies.

        Args:
            predictions: labels/predictions to decode back in their original string form
            label_vocabulary: label vocabulary, either for the tags of task A+B or for the categories of task C+D
            polarity_vocabulary: vocabulary of the polarities of task C+D
            mode: defines which task the labels are from, either "ab" or "cd"
        """

        decoded_predictions = list()

        if mode == "ab":
            for indices in predictions:
                # vocabulary integer to string used to obtain the corresponding label from the index
                decoded_predictions.append([label_vocabulary.get_itos()[i] for i in indices])
        elif mode == "cd":
            assert polarity_vocabulary is not None, "A valid polarity vocabulary is necessary for mode \"cd\""
            # for each predicted matrix (`n_categories` x `n_polarities+1` (5x5) shape)
            for preds_2d in predictions:
                # get the most probable polarity for each category,
                # where `preds[i]` is the polarity's index of the category of index `i`
                preds = torch.argmax(preds_2d, dim=-1)

                sample_predictions = list()
                for category_idx, polarity_idx in enumerate(preds):
                    # the last polarity index is to predict the categories the sample do not have,
                    # i.e. we only take the labels with the previous polarities
                    if polarity_idx != len(polarity_vocabulary):
                        sample_predictions.append((label_vocabulary.get_itos()[category_idx],
                                                   polarity_vocabulary.get_itos()[polarity_idx]))
                decoded_predictions.append(sample_predictions)
        else:
            raise Exception(f"\"{mode}\" is not a valid mode")

        return decoded_predictions

    def __preprocess_samples(self) -> None:
        """
        Preprocess the samples in the dataset by tokenizing the sentences, applying a POS tagging procedure and also
        tagging each token with the IOB+polarity schema.
        """

        samples = self.raw_samples
        tokens = list()
        pos_tags = list()
        tags = list()

        self.samples = {key: [dic[key] for dic in samples] for key in samples[0]}

        for i in range(len(samples)):
            sample_text = self.raw_samples[i]["text"]
            sample_tokens = self.tokenizer.tokenize(sample_text)
            sample_pos_tags = [pos[1] for pos in nltk.pos_tag(sample_tokens)]

            if not self.no_labels:
                sample_tags = self.__tag_sample(sample_text, self.raw_samples[i]["targets"], sample_tokens)
            else:
                sample_tags = [None for _ in range(len(sample_tokens))]

            tokens.append(sample_tokens)
            pos_tags.append(sample_pos_tags)
            tags.append(sample_tags)

        self.encoded_samples["pos_tags"] = pos_tags
        self.encoded_samples["tokens"] = tokens
        self.encoded_samples["tags"] = tags

        if self.has_categories and not self.no_labels:
            self.encoded_samples["categories"] = self.samples["categories"]

    def __tag_sample(self, sample_text: str, sample_targets: List, sample_tokens: List[str]) -> List[str]:
        """
        Parse the "targets" of each sample and returns a list of IOB+polarity tags (either "O" or "B|I-{polarity}"),
        one for each token in the sentence.

        Args:
            sample_text: the plain text of the sample to tag
            sample_targets: the plain targets list of the sample to tag
            sample_tokens: a consistent tokenization of the sentence
        """

        # initialize all tokens as Outside of an aspect term
        tags = ["O" for _ in range(len(sample_tokens))]

        # for each sample's target
        for target in sample_targets:
            if len(target) != 3: continue # fixes grader's missing polarity
            (start, end), aspect_term, polarity = target
            # tokenize the target aspect term and all that comes before in the sentence
            target_subwords_tokens = self.tokenizer.tokenize(sample_text[start:end])
            target_previous_tokens = self.tokenizer.tokenize(sample_text[:start])
            n_prev_tokens = len(target_previous_tokens)

            # compute a list of the indices (in terms of tokens) where the target aspect term is placed
            target_indices = list(range(n_prev_tokens, n_prev_tokens + len(target_subwords_tokens)))

            # tag with B (Beginning) the first token of the aspect term and with I (Inside) all the others
            for i, target_index in enumerate(target_indices):
                tags[target_index] = f"B-{polarity}" if i == 0 else f"I-{polarity}"

        return tags

    def __len__(self) -> int:
        return len(self.samples["text"])

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, List[str], str]]:

        return {
            "targets": self.samples["targets"][idx] if not self.no_labels else None,
            "text": self.samples["text"][idx],
            "tokens": self.encoded_samples["tokens"][idx],
            "pos_tags": self.encoded_samples["pos_tags"][idx],
            "tags": self.encoded_samples["tags"][idx],
            "categories": self.encoded_samples["categories"][idx] if self.has_categories and not self.no_labels else None,
            "token_idxs": self.__get_indices("token_idxs", idx),
            "pos_tag_idxs": self.__get_indices("pos_tag_idxs", idx),
            "tag_idxs": self.__get_indices("tag_idxs", idx) if not self.no_labels else None,
            "category_idxs": self.__get_indices("category_idxs", idx)
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
                 num_workers: int = 0,
                 pin_memory: bool = False,
                 has_category: bool = False) -> None:
        super().__init__()
        self.train_set = None
        self.val_set = None
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.tokenizer = tokenizer
        self.vocabs = vocabularies
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.has_category = has_category

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_set = ABSADataset(self.train_samples,
                                     tokenizer=self.tokenizer,
                                     vocabularies=self.vocabs,
                                     has_categories=self.has_category)

        self.val_set = ABSADataset(self.val_samples,
                                   tokenizer=self.tokenizer,
                                   vocabularies=self.vocabs,
                                   has_categories=self.has_category)

        if self.vocabs is None:
            self.vocabs = dict()
            self.vocabs["text"] = build_vocab(self.train_set + self.val_set)
            self.vocabs["pos"] = build_pos_vocab(self.train_set + self.val_set)
            self.vocabs["tag"] = build_label_vocabs(self.train_set)

            if self.has_category:
                self.vocabs["categories"], self.vocabs["category_polarities"] = build_category_vocabs(self.train_set)

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
