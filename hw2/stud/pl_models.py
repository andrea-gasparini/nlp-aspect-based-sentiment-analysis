from dataclasses import asdict, is_dataclass
from typing import *

import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor

from stud import utils
from stud.dataset import ABSADataset
from stud.models import AspectClassifier, HParams


class PlAspectClassifier(pl.LightningModule):

    def __init__(self, hparams: Union[HParams, Dict], embeddings: Optional[torch.FloatTensor] = None,
                 ignore_index: int = -100, *args, **kwargs) -> None:
        super().__init__()

        self.save_hyperparameters(asdict(hparams) if is_dataclass(hparams) else hparams)
        self.loss_function = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.model = AspectClassifier(self.hparams, embeddings)

        self.label_key = utils.get_label_key(self.hparams.mode) + "_idxs"

    def forward(self, batch) -> Dict[str, Tensor]:
        out = {"logits": self.model(batch)}

        if self.hparams.mode == "ab":
            out["preds"] = torch.argmax(out["logits"], dim=-1)
        elif self.hparams.mode == "cd":
            # reshape logits in the `n_categories` x `n_polarities+1` (5x5) encoding
            out["logits"] = out["logits"].reshape(-1, len(self.hparams.label_vocab), len(self.hparams.polarity_vocab)+1)
            out["preds"] = torch.softmax(out["logits"], dim=-1)

        return out

    def step(self, batch) -> Dict[str, Tensor]:
        """
        Generic step with the common operations to perform in all training, validation and test steps.
        Comprises invoking the forward pass (i.e. computing logits and predictions) and computing the loss.
        """
        out = self.forward(batch)
        logits = out["logits"]
        labels: Tensor = batch[self.label_key]

        if self.hparams.mode == "ab":
            # adapt logits and labels to fit the format required by the loss function
            logits = logits.view(-1, logits.shape[-1])
            labels = labels.view(-1)
            out["loss"] = self.loss_function(logits, labels)
        elif self.hparams.mode == "cd":
            # keep only the index of the correct polarity for each category instead of the one-hot encoding
            labels = torch.argmax(labels, dim=-1)

            n_categories = len(self.hparams.label_vocab)

            # compute the loss as the sum of the losses computed on each independent label
            out["loss"] = 0
            for category_idx in range(n_categories):
                category_logits, polarities = logits[:, category_idx], labels[:, category_idx]
                normalized_loss = 1 / n_categories * self.loss_function(category_logits, polarities)
                out["loss"] += normalized_loss

        return out

    def training_step(self, batch) -> Dict[str, Tensor]:
        out = self.step(batch)

        self.log("train_loss",
                 out["loss"],
                 prog_bar=True,
                 on_step=False,
                 on_epoch=True)

        return out

    def validation_step(self, batch) -> Dict[str, Union[Tensor, List[List[str]]]]:
        out = self.step(batch)

        self.log("valid_loss",
                 out["loss"],
                 prog_bar=True,
                 on_step=False,
                 on_epoch=True)

        # necessary in order to compute other metrics at the end of the epoch
        out["tokens"] = batch["tokens"]
        out["targets"] = batch["targets"]
        if self.hparams.mode == "cd":
            out["categories"] = batch["categories"]

        return out

    def validation_epoch_end(self, val_step_outputs: List[Dict[str, Union[Tensor, List[List[str]]]]]):
        """
        Computes and logs macro F1-scores at each epoch end for both the identification and the classification tasks.

        Args:
            val_step_outputs: list of outputs computed at each validation step
        """
        predictions = list()
        gold_labels = list()

        for out in val_step_outputs:

            if self.hparams.mode == "ab":
                gold_labels += [{"targets": targets} for targets in out["targets"]]
            elif self.hparams.mode == "cd":
                gold_labels += [{"categories": categories} for categories in out["categories"]]

            predictions += self.decode_predictions(out["tokens"], out["preds"])

        extraction_f1, evaluation_f1 = 0, 0

        if self.hparams.mode == "cd":
            extraction_f1 = utils.evaluate_sentiment(gold_labels, predictions, "Category Extraction")
            evaluation_f1 = utils.evaluate_sentiment(gold_labels, predictions, "Category Sentiment")
        elif self.hparams.mode == "ab":
            extraction_f1 = utils.evaluate_extraction(gold_labels, predictions)
            evaluation_f1 = utils.evaluate_sentiment(gold_labels, predictions, "Aspect Sentiment")

        self.log_dict({
            "valid_aspect_identification_f1": extraction_f1,
            "valid_aspect_polarity_classification_f1": evaluation_f1,
        })

    def test_step(self, batch):
        out = self.step(batch)

        self.log("test_loss",
                 out["loss"],
                 prog_bar=True,
                 on_step=False,
                 on_epoch=True)

    def predict(self, batch) -> List[Dict[str, List[Tuple[str, str]]]]:
        """
        Computes the predictions of the given batch and consequently decodes the result.
        """
        return self.decode_predictions(batch["tokens"], self(batch)["preds"])

    def decode_predictions(self, tokens: List[List[str]], preds: Tensor) -> List[Dict[str, List[Tuple[str, str]]]]:
        """
        Decodes the predictions in a format analogue to the one in the dataset and required by the homework's grader.

        Args:
            tokens: a batch of tokens associated to the predictions to decode
            preds: a batch of predictions to decode
        """

        decoded_preds = ABSADataset.decode_output(preds,
                                                  self.hparams.label_vocab,
                                                  self.hparams.polarity_vocab if self.hparams.mode == "cd" else None,
                                                  self.hparams.mode)

        if self.hparams.mode == "cd":
            return [{"categories": categories} for categories in decoded_preds]
        elif self.hparams.mode == "ab":

            out = list()

            # for each sample (tokens, preds) to decode
            for sample_tokens, sample_preds in zip(tokens, decoded_preds):

                targets = list()

                terms = list()
                polarities = list()

                # for each token and related prediction
                for token, pred in zip(sample_tokens, sample_preds):
                    # if the token is predicted as at the Beginning (B) or Inside (I) an aspect term
                    if pred[0] in ["B", "I"]:
                        # store the token and its predicted polarity
                        terms.append(token)
                        polarities.append(pred[2:])
                    # if we already have stored some tokens as a predicted aspect term
                    elif len(terms) != 0 and len(polarities) != 0:

                        # compute occurrences of each polarity
                        cnt_list = np.array(list(Counter(polarities).values()))

                        if len(set(cnt_list)) > 1 and np.all(cnt_list == cnt_list[0]):
                            # when there is the same amount of different polarities,
                            # e.g. ["positive", "positive", "negative", "negative"] --> "conflict"
                            polarity = "conflict"
                        else:
                            # otherwise take the polarity with the maximum amount of occurrences
                            polarity = max(polarities, key=polarities.count)

                        targets.append((" ".join(terms), polarity))

                        terms = list()
                        polarities = list()

                out.append({"targets": targets})

            return out

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
