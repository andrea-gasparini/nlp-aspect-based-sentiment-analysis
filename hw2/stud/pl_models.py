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

        return out
    def test_step(self, batch, batch_idx):
        out = self.step(batch, batch_idx)

        self.log("test_loss",
                 out["loss"],
                 prog_bar=True,
                 on_step=False,
                 on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())
