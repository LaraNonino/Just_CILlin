import pytorch_lightning as L
import torch

class MajorityVotingNet(L.LightningModule): # only performs inference!
    def __init__(
        self,
        datamodules: list,  
        nets: list,
        trainers: list,
    ):
        self.datamodules = datamodules # different datamodules for different tokenizers
        self.nets = nets
        self.trainers = trainers
        for dm, net, trainer, net in zip(self.datamodules, self.nets, self.trainers):
            # train each model
            dm.setup("fit")
            trainer.fit(model=net, datamodule=dm)
            # validate each model
            trainer.validate(model=net, datamodule=dm)
    
    def predict_step(self, batch, batch_idx):
        predictions = []
        for dm, net in zip(self.datamodules, self.nets):
            x = dm.tokenizer(batch)
            y_hat = net(x)
            predictions += [y_hat]
        predictions = torch.stack(predictions, dim=1).sum(dim=1)
        y_hat = torch.argmax(predictions, dim=-1)
        return torch.stack((batch["id"], y_hat), dim=1)
