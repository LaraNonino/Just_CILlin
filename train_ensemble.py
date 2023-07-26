import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelSummary, LearningRateMonitor, ModelCheckpoint, EarlyStopping

import torch
import torch.nn as nn

import os
import csv
from datetime import datetime

from dataset.twitter_dataset import TwitterDataModule
from recipes.sentiment_analysis import SentimentAnalysisNet
from recipes.majority_voting import MajorityVotingNet

def timestamp(format):
    ts = datetime.timestamp(datetime.now())
    return datetime.fromtimestamp(ts).strftime(format)

def save_predictions(predictions, file_name):
    with open(file_name, 'w+') as file:
        writer = csv.writer(file)
        writer.writerow(['Id', 'Prediction'])
        for i, pred in predictions:
            writer.writerow([i.item(), pred.item()])

def main():
    L.seed_everything(42, workers=True)
    batch_size = 256

    # Bert embeddings
    PRETRAINED_MODEL_NAME =  'distilroberta-base' #'distilbert-base-uncased'
    from transformers import AutoTokenizer

    print("prepearing model 1...")
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
    dm1 = TwitterDataModule(
        ["twitter-datasets/train_pos_full.txt", "twitter-datasets/train_neg_full.txt"],
        "twitter-datasets/test_data.txt",
        tokenizer=tokenizer,
        tokenizer_kwargs={
            "truncation": True,
            "padding": True,
        },
        batch_size=batch_size,
        num_workers=2,
        val_percentage=0.1
    )

    # Run datamodule to check input dimensions

    # 2. Models

    from models.transformer import BertUnpooledClassifier
    from models.attention import SelfAttention3D
    model1 = BertUnpooledClassifier(
        PRETRAINED_MODEL_NAME,
        classifier=nn.Sequential(
            SelfAttention3D(
                embed_dim=768, 
                q_dim=768,
                v_dim=256,
                collapse=True
            ),
            nn.Dropout(p=0.5),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(8, 2),
        )
    )
    net1 = SentimentAnalysisNet(
        model1,
        label_smoothing=0.1,
        lr=1e-5,
        # sched_step_size=3,
        # sched_gamma=0.2,
    )

    model2 = BertUnpooledClassifier(
        PRETRAINED_MODEL_NAME,
        classifier=nn.Sequential(
            SelfAttention3D(
                embed_dim=768, 
                q_dim=768,
                v_dim=256,
                collapse=True
            ),
            nn.Dropout(p=0.5),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(8, 2),
        )
    )
    net2 = SentimentAnalysisNet(
        model2,
        label_smoothing=0.1,
        lr=1e-5,
        # sched_step_size=3,
        # sched_gamma=0.2,
    )

    model3 = BertUnpooledClassifier(
        PRETRAINED_MODEL_NAME,
        classifier=nn.Sequential(
            SelfAttention3D(
                embed_dim=768, 
                q_dim=768,
                v_dim=256,
                collapse=True
            ),
            nn.Dropout(p=0.5),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(8, 2),
        )
    )
    net3 = SentimentAnalysisNet(
        model3,
        label_smoothing=0.1,
        lr=1e-5,
        # sched_step_size=3,
        # sched_gamma=0.2,
    )

    # 4. Train

    trainer1 = L.Trainer(
        max_epochs=1,
        callbacks=[
            ModelSummary(max_depth=5), 
            LearningRateMonitor(logging_interval='step'),
            ModelCheckpoint(save_top_k=3, monitor='val_loss'),
            EarlyStopping(monitor="val_loss", mode="min")
        ],
        deterministic=True, 
        log_every_n_steps=100,
        accelerator="gpu",
    )

    trainer2 = L.Trainer(
        max_epochs=1,
        callbacks=[
            ModelSummary(max_depth=5), 
            LearningRateMonitor(logging_interval='step'),
            ModelCheckpoint(save_top_k=3, monitor='val_loss'),
            EarlyStopping(monitor="val_loss", mode="min")
        ],
        deterministic=True, 
        log_every_n_steps=100,
        accelerator="gpu",
    )

    trainer3 = L.Trainer(
        max_epochs=1,
        callbacks=[
            ModelSummary(max_depth=5), 
            LearningRateMonitor(logging_interval='step'),
            ModelCheckpoint(save_top_k=3, monitor='val_loss'),
            EarlyStopping(monitor="val_loss", mode="min")
        ],
        deterministic=True, 
        log_every_n_steps=100,
        accelerator="gpu",
    )

    net = MajorityVotingNet(datamodules=[dm1, dm1, dm1], nets=[net1, net2, net3], trainers=[trainer1, trainer2, trainer3])
    trainer_ensemble = L.Trainer(
       accelerator="gpu"
    )

    predictions = trainer_ensemble.predict(model=net)
    path = 'predictions/{}'.format(timestamp('%d-%m-%Y-%H:%M:%S'))
    os.makedirs(path, exist_ok=True)
    save_predictions(torch.vstack(predictions), os.path.join(path, 'predictions.csv'))
    print("finished!")

if __name__ == "__main__":
    main()