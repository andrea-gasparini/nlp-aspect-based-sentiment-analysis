import pytorch_lightning as pl
import torch

from stud.constants import PAD_INDEX
from stud.models import AspectTermsClassifier


class PlAspectTermsClassifier(pl.LightningModule):
    def __init__(self, hparams, embeddings=None, ignore_index: int = PAD_INDEX):
        super().__init__()

        self.save_hyperparameters(hparams)
        self.loss_function = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.model = AspectTermsClassifier(self.hparams, embeddings)

        if self.hparams.mode == "a":
            self.label_key = "bio_idxs"
        elif self.hparams.mode == "b":
            self.label_key = "sentiment_idxs"
        elif self.hparams.mode == "ab":
            self.label_key = "tag_idxs"

    def forward(self, batch):
        logits = self.model(batch)
        predictions = torch.argmax(logits, -1)
        return logits, predictions

    def step(self, batch, batch_idx):
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
            "labels": labels,
            "preds": preds
        }

    def training_step(self, batch, batch_idx):
        out = self.step(batch, batch_idx)
        loss = out["loss"]

        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        out = self.step(batch, batch_idx)
        loss = out["loss"]

        self.log("valid_loss", loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        out = self.step(batch, batch_idx)
        loss = out["loss"]

        self.log("test_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())
