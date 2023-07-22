import csv, os
import torch
from datetime import datetime
from dataset.tw_data import TWBertDataModule
import pytorch_lightning as L
from models.crnn_bert import CRNNBertModel

N_EPOCHS = 1
LR_RATE = 2e-5

BATCH_SIZE = 256
N_WORKERS = 2

def main():
    L.seed_everything(42, workers=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))
    
    # Load and create datasets
    data_module = TWBertDataModule(path_predict="twitter-datasets/test_data.txt", batch_size=BATCH_SIZE, val_percentage=0.1, num_workers=N_WORKERS)
    data_module.setup('predict')
    
    # Validate
    model = CRNNBertModel.load_from_checkpoint("lightning_logs/crnn_param200/checkpoints/epoch=1-step=17580.ckpt", lr=LR_RATE)
    trainer = L.Trainer(max_epochs=N_EPOCHS, deterministic=True, log_every_n_steps=125, accelerator=device)
    predictions = trainer.predict(model, data_module.predict_dataloader())

    path = 'predictions/crnn_param200/'
    os.makedirs(path, exist_ok=True)
    save_predictions(torch.vstack(predictions), os.path.join(path, 'predictions.csv'))

def timestamp(format):
    ts = datetime.timestamp(datetime.now())
    return datetime.fromtimestamp(ts).strftime(format)

def save_predictions(preds, file_name):
    with open(file_name, 'w+') as file:
        writer = csv.writer(file)
        writer.writerow(['Id', 'Prediction'])
        for line in preds:
            pred = -1 if line[1].item() == 0 else 1
            writer.writerow([line[0].item(), pred])
            

if __name__ == "__main__":
    main()