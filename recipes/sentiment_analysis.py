import pytorch_lightning as L

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Any, Callable

class SentimentAnalysisNet(L.LightningModule):
    def __init__(
        self, 
        model: nn.Module,
        lr: float,
    ) -> None:
        super().__init__()
        self.model = model
        self.lr = lr

    def forward(self, x) -> Any:
        print(self.model)
        return self.model(x.float())
    
    def training_step(self, batch, batch_idx):
        x, y = batch[:, :-1], batch[:, -1]
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        print(loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    