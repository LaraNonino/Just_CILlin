import pytorch_lightning as L

import torch
import torch.nn as nn
import torchmetrics

class SentimentAnalysisNet(L.LightningModule):
    def __init__(
        self, 
        model: nn.Module,
        label_smoothing: float=0,
        lr: float=2e-5,
        sched_step_size: int=None,
        sched_gamma: float=None,
    ):
        super().__init__()
        self.model = model
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.val_accuracy = torchmetrics.Accuracy(task="binary", threshold=0.5)
        self.train_accuracy = torchmetrics.Accuracy(task="binary", threshold=0.5)
        
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.scheduler = None
        if sched_step_size and sched_gamma:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=sched_step_size, gamma=sched_gamma
            )

    def forward(self, x):
        y_hat = self.model(x)
        return y_hat
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = self.criterion(y_hat, y)
        self.train_accuracy.update(torch.argmax(y_hat, dim=-1), y)

        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train_acc", self.train_accuracy.compute())

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = self.criterion(y_hat, y)
        self.val_accuracy.update(torch.argmax(y_hat, dim=-1), y)

        self.log("val_loss", loss)
        self.log("val_acc", self.val_accuracy.compute())

    def predict_step(self, batch, batch_idx):
        y_hat = torch.argmax(self(batch), dim=-1)
        y_hat = torch.where(y_hat == 1, 1, -1) # pos: +1, neg: -1

        return torch.stack((batch["id"], y_hat), dim=1)
    
    def configure_optimizers(self):
        if self.scheduler is not None:
            return [self.optimizer], [self.scheduler]
        return [self.optimizer]