import os
import torch
from datetime import datetime
from dataset.tw_data import TWBertDataModule
import pytorch_lightning as L
from models.simple_bert import SABertModel
from models.rnn_bert import RNNBertModel

N_EPOCHS = 1
LR_RATE = 2e-5

BATCH_SIZE = 16
N_WORKERS = 12

def main():
    ts = timestamp('%d-%m-%Y-%H:%M:%S')
    L.seed_everything(42, workers=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))
    
    # Load and create datasets
    data_module = TWBertDataModule("twitter-datasets/train_pos.txt", "twitter-datasets/train_neg.txt", 
                                   batch_size=BATCH_SIZE, val_percentage=0.1, num_workers=N_WORKERS)
    data_module.setup('fit')
    
    # Training
    # model = SABertModel(lr=LR_RATE)
    model = RNNBertModel(lr=LR_RATE)

    trainer = L.Trainer(max_epochs=N_EPOCHS, deterministic=True, log_every_n_steps=125, accelerator=device)
    trainer.fit(model, data_module.train_dataloader(), data_module.val_dataloader())
    trainer.validate(model, data_module.val_dataloader())
    
    path = 'out/models/{}'.format(ts)
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path, '{}.pt'.format(ts)))

def timestamp(format):
    ts = datetime.timestamp(datetime.now())
    return datetime.fromtimestamp(ts).strftime(format)

if __name__ == "__main__":
    main()