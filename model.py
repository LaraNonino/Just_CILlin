import csv
import os
from datetime import datetime

import pytorch_lightning as L
import torch
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, ModelSummary
from transformers import AutoTokenizer

from dataset.twitter_dataset import TwitterDataModule
from models.bert import CRNNBertModel
from recipes.sentiment_analysis import SentimentAnalysisNet

N_EPOCHS = 5
LR_RATE = 2e-5

BATCH_SIZE = 256
N_WORKERS = 2

PRETRAINED_MODEL_NAME = 'distilbert-base-uncased'

def main():
    L.seed_everything(42, workers=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))

    # 1. Dataset
    # Use BERT transformer embeddings

    print("Preparing data module...")

    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
    dm = TwitterDataModule(
        ["twitter-datasets/train_pos_full.txt", "twitter-datasets/train_neg_full.txt"],
        "twitter-datasets/test_data.txt",
        tokenizer=tokenizer,
        tokenizer_kwargs={"truncation": True, "padding": True},
        batch_size=BATCH_SIZE,
        num_workers=N_WORKERS,
        val_percentage=0.1
    )

    # Run datamodule to check input dimensions
    dm.setup(stage="fit")
    print("Data module set up.")

    # 2. Model

    model = CRNNBertModel(pretrained_model_name=PRETRAINED_MODEL_NAME)

    # 3. Lightning net

    net = SentimentAnalysisNet(
        model,
        # label_smoothing=0.1,
        lr=LR_RATE,
        # sched_step_size=3,
        # sched_gamma=0.2,
    )

    # 4. Train

    trainer = L.Trainer(
        max_epochs=N_EPOCHS,
        callbacks=[
            ModelSummary(max_depth=3), 
            # LearningRateMonitor(logging_interval='step'),
            ModelCheckpoint(save_top_k=3, monitor='val_loss'),
            # EarlyStopping(monitor="val_loss", mode="min")
        ],
        deterministic=True, 
        log_every_n_steps=100,
        accelerator=device,
    )

    print("Start training...")
    trainer.fit(net, dm.train_dataloader(), dm.val_dataloader())
    print("Finished training")
    trainer.validate(net, dm.val_dataloader())

    # 5. Predict

    # dm.setup(stage="predict")

    # # net = SentimentAnalysisNet.load_from_checkpoint(
    # #     "lightning_logs/version_22136887/checkpoints/epoch=3-step=140628.ckpt",
    # #     model,
    # #     # lr=2e-5,
    # #     # sched_step_size=1, # every half an epoch 
    # #     # sched_gamma=0.5,
    # # )

    # print("Start prediction...")
    # predictions = trainer.predict(net, dm.predict_dataloader())
    # path = 'predictions/{}'.format(timestamp('%d-%m-%Y-%H:%M:%S'))
    # os.makedirs(path, exist_ok=True)
    # save_predictions(torch.vstack(predictions), os.path.join(path, 'predictions.csv'))
    
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
    main()