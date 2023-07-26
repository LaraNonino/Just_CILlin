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

    # Model 1
    print("prepearing model 1...")

    model_name1 =  'distilbert-base-uncased' # 'distilroberta-base'
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name1)
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

    from models.transformer import BertUnpooledClassifier
    from models.attention import SelfAttention3D

    model1 = BertUnpooledClassifier(
        model_name1,
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
        lr=2e-5,
        sched_step_size=1,
        sched_gamma=0.1,
    )

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

    # Model 2
    print("prepearing model 2...")
    from models.transformer import TransformerClassifier

    model_name2 =  'distilroberta-base'

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

    model2 = TransformerClassifier(
        model_name2,
        model_kwargs={
            "hidden_dropout_prob": 0.1, # "dropout" (BertConfig); "hidden_dropout_prob" (RobertaConfig)
            "attention_probs_dropout_prob": 0.1,
        }
    )

    net2 = SentimentAnalysisNet(
        model2,
        label_smoothing=0.1,
        lr=1e-5,
        sched_step_size=1,
        sched_gamma=0.1,
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

    # Majority voting (inference)
    dm = TwitterDataModule(
        ["twitter-datasets/train_pos_full.txt", "twitter-datasets/train_neg_full.txt"],
        "twitter-datasets/test_data.txt",
    )
    dm.setup("predict") # only setup predict data
    net = MajorityVotingNet(datamodules=[dm1, dm2], nets=[net1, net2], trainers=[trainer1, trainer2])
    trainer_ensemble = L.Trainer(accelerator="gpu")
    predictions = trainer_ensemble.predict(net, dm.predict_dataloader())

    path = 'predictions/{}'.format(timestamp('%d-%m-%Y-%H:%M:%S'))
    os.makedirs(path, exist_ok=True)
    save_predictions(torch.vstack(predictions), os.path.join(path, 'predictions.csv'))
    print("finished!")

if __name__ == "__main__":
    main()