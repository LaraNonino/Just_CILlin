import pytorch_lightning as L
import torch
import torch.nn as nn

class MajorityVotingNet(L.LightningModule): # only performs inference!
    def __init__(
        self,
        tokenizers: list,
        nets: nn.ModuleList,
    ):
        super().__init__()
        self.tokenizers = tokenizers
        self.nets = nets

    def forward(self, x):
        predictions = torch.zeros((len(x), 2)) 
        for t, net in zip(self.tokenizers, self.nets):
            x = t(x)
            y_hat = net(x)
            predictions += y_hat
        return predictions
    
    def predict_step(self, batch, batch_idx):
        predictions = torch.argmax(self(batch), dim=-1)
        return torch.stack((batch["id"], predictions), dim=1)
