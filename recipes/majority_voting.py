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

    @torch.inference_mode()
    def forward(self, x):
        predictions = torch.zeros((len(x), 2)).cuda()
        for t, net in zip(self.tokenizers, self.nets):
            X = t(x) # tokenize
            X = {key: (torch.tensor(val)).cuda() for key, val in X.items()} # create tensor data for forward pass and pass to cuda
            y_hat = net(X) # predict
            predictions += y_hat
        return predictions
    
    def predict_step(self, batch, batch_idx):
        y_hat = torch.argmax(self(batch), dim=-1)
        y_hat = torch.where(y_hat >= 0.5, 1, -1)
        ids = range(batch_idx * len(batch) + 1, (batch_idx+1) * len(batch) + 1) # ids from 1 to 10000
        return torch.stack((torch.tensor(ids).cuda(), y_hat), dim=1)
