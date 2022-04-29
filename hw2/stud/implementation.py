import os
from typing import List, Dict

import torch
from nltk import TreebankWordTokenizer
from torch.utils.data import DataLoader

from model import Model
from stud import utils
from stud.dataset import ABSADataset, padding_collate_fn
from stud.pl_models import PlAspectClassifier


def build_model_b(device: str) -> Model:
    """
    The implementation of this function is MANDATORY.
    Args:
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements aspect sentiment analysis of the ABSA pipeline.
            b: Aspect sentiment analysis.
    """
    return StudentModel(mode="b", device=device)


def build_model_ab(device: str) -> Model:
    """
    The implementation of this function is MANDATORY.
    Args:
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements both aspect identification and sentiment analysis of the ABSA pipeline.
            a: Aspect identification.
            b: Aspect sentiment analysis.

    """
    return StudentModel(mode="ab", device=device)


def build_model_cd(device: str) -> Model:
    """
    The implementation of this function is OPTIONAL.
    Args:
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements both aspect identification and sentiment analysis of the ABSA pipeline 
        as well as Category identification and sentiment analysis.
            c: Category identification.
            d: Category sentiment analysis.
    """
    return StudentModel(mode="cd", device=device)


class StudentModel(Model):

    def __init__(self, mode: str, device: str) -> None:

        self.mode = mode

        # load pre-computed vocabularies
        self.vocabularies = {
            "text": torch.load(f"model/vocabularies/text{'_restaurants' if mode == 'cd' else ''}.pt"),
            "tag": torch.load("model/vocabularies/iob_polarity.pt"),
            "pos": torch.load("model/vocabularies/pos.pt"),
            "categories": torch.load("model/vocabularies/category.pt"),
            "category_polarities": torch.load("model/vocabularies/polarity.pt")
        }

        # load NLTK stuff
        utils.nltk_downloads()

        # load GloVe embeddings
        self.glove_embeddings = utils.load_pretrained_embeddings("glove.6B.300d.txt",
                                                                 "model/embeddings/glove/",
                                                                 self.vocabularies["text"])

        self.tokenizer = TreebankWordTokenizer()

        if self.mode in ["ab", "b"]:
            checkpoint_name = "model_A+B.ckpt"
        elif self.mode == "cd":
            checkpoint_name = "model_C+D.ckpt"
        else:
            raise Exception(f"\"{self.mode}\" is not a valid mode")

        self.model = PlAspectClassifier.load_from_checkpoint(f"model/{checkpoint_name}",
                                                             map_location=device,
                                                             bert_model_name_or_path="model/bert-base-cased")
        self.model.freeze()

    def predict(self, samples: List[Dict]) -> List[Dict]:

        if self.mode == "cd":
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

        dataset = ABSADataset(samples,
                              tokenizer=self.tokenizer,
                              vocabularies=self.vocabularies,
                              has_categories=self.mode == "cd",
							  no_labels=self.mode != "b")

        dataloader = DataLoader(dataset,
                                shuffle=False,
                                batch_size=self.model.hparams.batch_size,
                                collate_fn=padding_collate_fn)

        predictions = list()
        for batch in dataloader:
            predictions += self.model.predict(batch)

        return predictions
