import pytorch_lightning as pl
import torch

from stud import utils
from stud.constants import PAD_INDEX
from stud.models import AspectTermsClassifier
from typing import *


class PlAspectTermsClassifier(pl.LightningModule):
    def __init__(self, hparams, embeddings=None, ignore_index: int = PAD_INDEX, *args, **kwargs) -> None:
        super().__init__()

        self.save_hyperparameters(hparams)
        self.loss_function = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.model = AspectTermsClassifier(self.hparams, embeddings)

        self.label_key = utils.get_label_key(self.hparams.mode) + "_idxs"

    def forward(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.model(batch)
        predictions = torch.argmax(logits, -1)
        return logits, predictions

    def step(self, batch, batch_idx) -> Dict[str, torch.Tensor]:
        labels = batch[self.label_key]
        # We receive one batch of data and perform a forward pass:
        logits, preds = self.forward(batch)
        # Adapt logits and labels to fit the format required by the loss function
        logits = logits.view(-1, logits.shape[-1])
        labels = labels.view(-1)

        loss = self.loss_function(logits, labels)

        return {
            "loss": loss,
            "logits": logits,
            "labels": batch[self.label_key],
            "preds": preds
        }

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

        return out

    def validation_epoch_end(self, val_step_outputs: List[Dict[str, torch.Tensor]]):
        predictions = list()
        gold_targets = list()

        for out in val_step_outputs:
            gold_targets += [{"targets": targets} for targets in out["targets"]]
            predictions += self.decode_predictions(out["tokens"],
                                                   out["labels"],
                                                   out["preds"])            

        extraction_f1, evaluation_f1 = 0, 0

        try:
            extraction_f1 = evaluate_extraction(gold_targets, predictions)
        except ZeroDivisionError:
            pass

        try:
            evaluation_scores = evaluate_sentiment(gold_targets,
                                                   predictions,
                                                   "Aspect Sentiment")
            evaluation_f1 = evaluation_scores["Macro_f1"]
        except ZeroDivisionError:
            pass

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

        decoded_preds = ABSADataset.decode_output(predictions_batch, self.hparams.label_vocab)

        mode = self.hparams.mode

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

                    if tags[0] == "I": break

                    if "b" in mode:
                        polarities = [x for x in polarities if x != ""]

                    if len(polarities) != 0:

                        polarity = None

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
