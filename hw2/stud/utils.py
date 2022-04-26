import evaluate
import io
import os
import torch

from contextlib import redirect_stdout
from torchtext.vocab import Vectors, Vocab
from typing import *


def load_pretrained_embeddings(filename: str, cache_dir: str, vocab: Vocab) -> torch.Tensor:
    """
	Loads from a local file static word embedding vectors (e.g. GloVe, FastText)
	and pairs them with the tokens contain in the given vocabulary.

    Args:
            filename: name of the file that contains the vectors
            cache_dir: directory for cached vectors
            vocab: vocabulary of tokens to be embedded
    """
    pretrained_embeddings = Vectors(filename, cache=cache_dir)
    embeddings = torch.randn(len(vocab), pretrained_embeddings.dim)
    initialised = 0

    for i, token in enumerate(vocab.get_itos()):
        token = token.lower()
        if token in pretrained_embeddings.stoi:
            initialised += 1
            embedding = pretrained_embeddings.get_vecs_by_tokens(token)
            embeddings[i] = embedding

    embeddings[vocab["[PAD]"]] = torch.zeros(pretrained_embeddings.dim)

    print(f"initialised {initialised} embeddings")
    print(f"randomly initialised {len(vocab) - initialised} embeddings")

    return embeddings


def get_pretrained_model(pretrained_model_name_or_path: str) -> str:
    """
    Returns the HuggingFace model name or the path to its local directory in case
    the given arg is a valid one.
    """
    return (pretrained_model_name_or_path
            if os.path.exists(pretrained_model_name_or_path)
            else os.path.basename(os.path.normpath(pretrained_model_name_or_path)))


def get_label_key(mode: str = "ab") -> str:
    return "tag" if mode == "ab" else "sentiment" if mode == "b" else "bio"


def evaluate_extraction(samples, predictions) -> float:
    scores = {"tp": 0, "fp": 0, "fn": 0}
    for label, pred in zip(samples, predictions):
        pred_terms = {term_pred[0] for term_pred in pred["targets"]}
        gt_terms = {term_gt[1] for term_gt in label["targets"]}

        scores["tp"] += len(pred_terms & gt_terms)
        scores["fp"] += len(pred_terms - gt_terms)
        scores["fn"] += len(gt_terms - pred_terms)

    precision = 100 * scores["tp"] / (scores["tp"] + scores["fp"])
    recall = 100 * scores["tp"] / (scores["tp"] + scores["fn"])
    f1 = 2 * precision * recall / (precision + recall)

    return f1


def evaluate_sentiment(samples, predictions, mode="Aspect Sentiment") -> Dict[str, float]:
    with redirect_stdout(io.StringIO()):
        return evaluate.evaluate_sentiment(samples, predictions, mode)[0]["ALL"]
