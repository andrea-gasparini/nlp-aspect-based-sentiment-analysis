from typing import *

import numpy as np
import pytorch_lightning as pl
import torch

from stud import utils
from stud.dataset import ABSADataset
from stud.models import AspectTermsClassifier


class PlAspectTermsClassifier(pl.LightningModule):
    def __init__(self, hparams, embeddings=None, ignore_index: int = -100, *args, **kwargs) -> None:
        super().__init__()

        self.save_hyperparameters(hparams)
        self.loss_function = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.model = AspectTermsClassifier(self.hparams, embeddings)

        self.label_key = utils.get_label_key(self.hparams.mode) + "_idxs"

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        out = {"logits": self.model(batch)}

        if self.hparams.mode in ["ab", "a", "b"]:
            out["preds"] = torch.argmax(out["logits"], dim=-1)
        elif self.hparams.mode == "cd":
            out["preds"] = torch.softmax(out["logits"], dim=-1)

        return out

    def step(self, batch, batch_idx) -> Dict[str, torch.Tensor]:
        out = self.forward(batch)
        logits = out["logits"]
        labels: torch.Tensor = batch[self.label_key]
        out["labels"] = labels

        if self.hparams.mode in ["ab", "a", "b"]:
            # adapt logits and labels to fit the format required by the loss function
            logits = logits.view(-1, logits.shape[-1])
            labels = labels.view(-1)
            out["loss"] = self.loss_function(logits, labels)
        elif self.hparams.mode == "cd":
            # adapt logits to fit the format required by the loss function for each independent category+polarity,
            # keeping only the index of the correct polarity for each category instead of the one-hot encoding
            n_categories = len(self.hparams.label_vocab)
            matrix_shape = (-1, n_categories, len(self.hparams.polarity_vocab) + 1)
            logits = logits.reshape(matrix_shape)
            labels = torch.argmax(labels.reshape(matrix_shape), dim=-1)

            # compute the loss as the sum of the losses computed on each independent label
            out["loss"] = 0
            for category_idx in range(n_categories):
                category_logits, polarities = logits[:, category_idx], labels[:, category_idx]
                normalized_loss = 1 / n_categories * self.loss_function(category_logits, polarities)
                out["loss"] += normalized_loss

        return out

    def training_step(self, batch, batch_idx) -> Dict[str, torch.Tensor]:
        out = self.step(batch, batch_idx)

        self.log("train_loss",
                 out["loss"],
                 prog_bar=True,
                 on_step=False,
                 on_epoch=True)

        return out

    def validation_step(self, batch, batch_idx) -> Dict[str, torch.Tensor]:
        out = self.step(batch, batch_idx)

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

    def validation_epoch_end(self, val_step_outputs: List[Dict[str, Union[torch.Tensor, List[List[str]]]]]):
        predictions = list()
        gold_targets = list()

        for out in val_step_outputs:
            if self.hparams.mode in ["ab", "a", "b"]:
                gold_targets += [{"targets": targets} for targets in out["targets"]]
            elif self.hparams.mode == "cd":
                gold_targets += [{"categories": categories} for categories in out["categories"]]
            predictions += self.decode_predictions(out["tokens"],
                                                   out["labels"],
                                                   out["preds"])

        if self.hparams.mode == "cd":
            extraction_f1 = utils.evaluate_sentiment(gold_targets, predictions, "Category Extraction")
            evaluation_f1 = utils.evaluate_sentiment(gold_targets, predictions, "Category Sentiment")
        else:
            extraction_f1 = utils.evaluate_extraction(gold_targets, predictions)
            evaluation_f1 = utils.evaluate_sentiment(gold_targets, predictions, "Aspect Sentiment")

        self.log_dict({
            "valid_aspect_sentiment_extraction_f1": extraction_f1,
            "valid_aspect_sentiment_evaluation_f1": evaluation_f1,
        })

    def test_step(self, batch, batch_idx):
        out = self.step(batch, batch_idx)

        self.log("test_loss",
                 out["loss"],
                 prog_bar=True,
                 on_step=False,
                 on_epoch=True)

    def decode_predictions(self,
                           tokens_batch: List[List[str]],
                           gold_labels_batch: torch.Tensor,
                           predictions_batch: torch.Tensor
                           ) -> List[Dict[str, List[Tuple[str, str]]]]:

        decoded_preds = ABSADataset.decode_output(predictions_batch,
                                                  self.hparams.label_vocab,
                                                  self.hparams.polarity_vocab if self.hparams.mode == "cd" else None,
                                                  self.hparams.mode)

        mode = self.hparams.mode

        if mode == "cd":
            return [{"categories": categories} for categories in decoded_preds]

        out = list()

        for tokens, gold_labels, preds in zip(tokens_batch, gold_labels_batch, decoded_preds):

            targets = list()

            terms = list()
            tags = list()
            polarities = list()

            for token, gold_label, pred in zip(tokens, gold_labels, preds):
                bio = (pred if "a" in mode else gold_label)[0]
                if bio == "I" or bio == "B":
                    terms.append(token)
                    tags.append(bio)
                    polarity = pred if "b" == mode else pred[2:] if "ab" == mode else ""
                    polarities.append(polarity)
                elif len(terms) != 0 and len(polarities) != 0:

                    if "b" in mode:
                        polarities = [x for x in polarities if x != ""]

                    if len(polarities) != 0:

                        cnt_list = np.array(list(Counter(polarities).values()))

                        if len(set(cnt_list)) > 1 and np.all(cnt_list == cnt_list[0]):
                            # when there is the same amount of different polarities,
                            # e.g. ["positive", "positive", "negative", "negative"] --> "conflict"
                            polarity = "conflict"
                        else:
                            # otherwise take the polarity with the maximum amount of occurrences
                            polarity = max(polarities, key=polarities.count)

                        targets.append((" ".join(terms), polarity))

                    tags = list()
                    terms = list()
                    polarities = list()

            out.append({"targets": targets})

        return out

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())
