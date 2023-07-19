import pytorch_lightning as L

import torch
import torch.nn as nn
import torchmetrics

class SentimentAnalysisNet(L.LightningModule):
    def __init__(
        self, 
        model: nn.Module,
        criterion: nn.Module=nn.BCELoss(),
        lr: float=10e-3,
        sched_step_size: int=None,
        sched_gamma: float=None,
    ):
        super().__init__()

        self.model = model
        self.criterion = criterion
        self.val_accuracy = torchmetrics.Accuracy(task="binary", threshold=0.5)
        # self.val_confmatrix
        
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.scheduler = None
        if sched_step_size and sched_gamma:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=sched_step_size, gamma=sched_gamma
            )

    def forward(self, x):
        output = self.model(x)
        return output
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.val_accuracy.update(y_hat, y)
        self.log("val_loss", loss)
        self.log("val_acc", self.val_accuracy.compute())
    
    def configure_optimizers(self):
        if self.scheduler is not None:
            return [self.optimizer], [self.scheduler]
        return [self.optimizer]