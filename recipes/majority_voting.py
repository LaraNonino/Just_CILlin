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
        self.nets = nets # trained nn.Modules

    def forward(self, x):
        predictions = torch.zeros((len(x), 2)) 
        for t, net in zip(self.tokenizers, self.nets):
            X = t(x) # tokenize
            X = {key: torch.tensor(val) for key, val in X.items()} # create tensor data for forward pass
            y_hat = net(X) # predict
            predictions += y_hat
        return predictions
    
    def predict_step(self, batch, batch_idx):
        predictions = torch.argmax(self(batch), dim=-1)
        ids = range(batch_idx * len(batch) + 1, (batch_idx+1) * len(batch) + 1) # ids from 1 to 10000
        return torch.stack((torch.tensor(ids, predictions), dim=1)
