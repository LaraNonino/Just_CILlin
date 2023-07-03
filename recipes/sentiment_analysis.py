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
        self.accuracy = torchmetrics.Accuracy(task="binary")
        
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.scheduler = None
        if sched_step_size and sched_gamma:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=sched_step_size, gamma=sched_gamma
            )

    def forward(self, x):
        output = self.model(x)
        if isinstance(output, tuple): # in RNNs
            output = output[0]
        return output
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        return loss
    
    def configure_optimizers(self):
        if self.scheduler is not None:
            return [self.optimizer], [self.scheduler]
        return [self.optimizer]
    