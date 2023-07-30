import csv
import os
from datetime import datetime
from typing import Sequence

import pytorch_lightning as L
import torch
from absl import app, flags
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, ModelSummary
from transformers import AutoTokenizer

from dataset.twitter_dataset import TwitterDataModule
from models.bert import CRNNBertModel
from recipes.sentiment_analysis import SentimentAnalysisNet

FLAGS = flags.FLAGS
flags.DEFINE_integer('n_epochs', 5, 'Number of epochs to train.')
flags.DEFINE_integer('batch_size', 256, 'Batch size.')
flags.DEFINE_integer('n_workers', 2, 'Numer of workers.')
flags.DEFINE_string('model', 'lightning_logs/crnn/checkpoints/epoch=2-step=26370.ckpt', 'Model to use for prediction.')

PRETRAINED_MODEL_NAME = 'distilbert-base-uncased'

def main(argv: Sequence[str]):
    L.seed_everything(42, workers=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))

    # 1. Dataset
    print("Preparing data module...")

    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
    dm = TwitterDataModule(
        path_predict="twitter-datasets/test_data.txt",
        tokenizer=tokenizer,
        tokenizer_kwargs={"truncation": True, "padding": True},
        batch_size=FLAGS.batch_size,
        num_workers=FLAGS.n_workers
    )

    dm.setup(stage="predict")
    print("Data module set up.")

    # 2. Load model
    net = SentimentAnalysisNet.load_from_checkpoint(checkpoint_path=FLAGS.model, model=CRNNBertModel())

    # 3. Predict
    trainer = L.Trainer(
        max_epochs=FLAGS.n_epochs,
        callbacks=[
            ModelSummary(max_depth=3), 
            LearningRateMonitor(logging_interval='step'),
            ModelCheckpoint(save_top_k=3, monitor='val_loss'),
            EarlyStopping(monitor="val_loss", mode="min")
        ],
        deterministic=True, 
        log_every_n_steps=100,
        accelerator=device,
        logger=False
    )

    print("Start prediction...")
    predictions = trainer.predict(net, dm.predict_dataloader())
    print("Finished prediction")

    # 4. Save predictions
    path = 'predictions/{}'.format(timestamp('%d-%m-%Y-%H:%M:%S'))
    os.makedirs(path, exist_ok=True)
    save_predictions(torch.vstack(predictions), os.path.join(path, 'predictions.csv'))
    
    print("Finished!")

def timestamp(format):
    ts = datetime.timestamp(datetime.now())
    return datetime.fromtimestamp(ts).strftime(format)

def save_predictions(predictions, file_name):
    with open(file_name, 'w+') as file:
        writer = csv.writer(file)
        writer.writerow(['Id', 'Prediction'])
        for i, pred in predictions:
            writer.writerow([i.item(), pred.item()])

if __name__ == "__main__":
    app.run(main)