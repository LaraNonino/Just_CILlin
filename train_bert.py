import torch
from dataset.tw_data import TWBertDataModule
import pytorch_lightning as L
from models.crnn_bert import CRNNBertModel

N_EPOCHS = 5
LR_RATE = 2e-5

BATCH_SIZE = 256
N_WORKERS = 2

def main():
    L.seed_everything(42, workers=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))
    
    # Load and create datasets
    data_module = TWBertDataModule("twitter-datasets/train_pos_full.txt", "twitter-datasets/train_neg_full.txt", 
                                   batch_size=BATCH_SIZE, val_percentage=0.1, num_workers=N_WORKERS)
    data_module.setup('fit')
    
    # Training
    model = CRNNBertModel(lr=LR_RATE, sched_step_size=1, sched_gamma=0.1)

    trainer = L.Trainer(max_epochs=N_EPOCHS, deterministic=True, log_every_n_steps=125, 
                        accelerator=device, callbacks=[L.callbacks.ModelCheckpoint(save_top_k=5, monitor='val_loss')])
    trainer.fit(model, data_module.train_dataloader(), data_module.val_dataloader())
    trainer.validate(model, data_module.val_dataloader())

if __name__ == "__main__":
    main()