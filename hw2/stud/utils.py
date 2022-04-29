import io
import os
from contextlib import redirect_stdout
from typing import Optional

import nltk
import torch
from torchtext.vocab import Vectors, Vocab

import evaluate
from stud import constants as const


def nltk_downloads(download_dir: Optional[str] = None) -> None:
    nltk.download('averaged_perceptron_tagger', download_dir=download_dir)


def load_pretrained_embeddings(filename: str, cache_dir: str, vocab: Vocab) -> torch.Tensor:
    """
    Loads from a local file static word embedding vectors (e.g. GloVe, FastText)
    and pairs them with the tokens contained in the given vocabulary.

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

    embeddings[vocab[const.PAD_TOKEN]] = torch.zeros(pretrained_embeddings.dim)

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
    if mode == "ab":
        return "tag"
    elif mode == "cd":
        return "category"
    else:
        raise Exception(f"\"{mode}\" is not a valid mode")


def evaluate_extraction(samples, predictions) -> float:
    tp, fp, fn = 0, 0, 0
    for label, pred in zip(samples, predictions):
        pred_terms = {term_pred[0] for term_pred in pred["targets"]}
        gt_terms = {term_gt[1] for term_gt in label["targets"]}

        tp += len(pred_terms & gt_terms)
        fp += len(pred_terms - gt_terms)
        fn += len(gt_terms - pred_terms)

    try:
        precision = 100 * tp / (tp + fp)
        recall = 100 * tp / (tp + fn)
        return 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        return 0


def evaluate_sentiment(samples, predictions, mode="Aspect Sentiment") -> float:
    with redirect_stdout(io.StringIO()):
        try:
            return evaluate.evaluate_sentiment(samples, predictions, mode)[0]["ALL"]["Macro_f1"]
        except ZeroDivisionError:
            return 0


def get_device(tensor: torch.Tensor) -> str:
    return "cuda" if tensor.is_cuda else "cpu"
