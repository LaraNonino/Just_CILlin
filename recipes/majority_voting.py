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
        print("fwd!")
        print(f"x: {x[0]}")
        predictions = torch.zeros((len(x), 2)) 
        for t, net in zip(self.tokenizers, self.nets):
            print(f"x: {len(x)}")
            x = t(x)
            y_hat = net(x)
            print(f"y_hat: {y_hat.shape}")
            predictions += y_hat
        return predictions
    
    def predict_step(self, batch, batch_idx):
        predictions = torch.argmax(self(batch), dim=-1)
        return torch.stack((batch["id"], predictions), dim=1)
