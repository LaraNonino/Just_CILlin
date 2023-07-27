import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelSummary, LearningRateMonitor, ModelCheckpoint, EarlyStopping

import torch
import torch.nn as nn
from transformers import AutoTokenizer, DataCollatorWithPadding
from peft import PromptTuningConfig, PrefixTuningConfig, PromptEncoderConfig

import os
import csv
from datetime import datetime
from functools import partial

from dataset.twitter_dataset import TwitterDataModule
from models.p_tuning import PTunedlassifier
from recipes.sentiment_analysis import SentimentAnalysisNet

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

    model_name = 'vinai/bertweet-base'
    if any(k in model_name for k in ("gpt", "opt", "bloom")):
        padding_side = "left"
    else:
        padding_side = "right"

    # Datamodule
    print("prepearing datamodule...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side=padding_side)
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer_kwargs = {
        "truncation": True,
        "max_length": None,
    }
    dm = TwitterDataModule(
        ["twitter-datasets/train_pos_full.txt", "twitter-datasets/train_neg_full.txt"],
        "twitter-datasets/test_data.txt",
        tokenizer=tokenizer,
        tokenizer_kwargs=tokenizer_kwargs,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, padding="longest"), # partial(tokenizer.pad, padding="longest", return_tensors="pt"),
        batch_size=batch_size,
        num_workers=2,
        val_percentage=0.1
    )
    dm.setup("fit")
    print("data module set up.")

    # Model
    peft_config = PromptTuningConfig(
        task_type="SEQ_CLS", 
        num_virtual_tokens=10
    )
    model = PTunedlassifier(
        model_name,
        peft_config
    )

    # Lightning module
    net = SentimentAnalysisNet(
        model,
        label_smoothing=0.05,
        lr=1e-3,
        # sched_step_size=1,
        # sched_gamma=0.1
    )

    trainer = L.Trainer(
        max_epochs=20,
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
    print("start training...")
    trainer.fit(model=net, datamodule=dm)
    trainer.validate(net, dm.val_dataloader())

    print("start prediction...")
    predictions = trainer.predict(net, dm.predict_dataloader())
    path = 'predictions/{}'.format(timestamp('%d-%m-%Y-%H:%M:%S'))
    os.makedirs(path, exist_ok=True)
    save_predictions(torch.vstack(predictions), os.path.join(path, 'predictions.csv'))
    print("finished!")


if __name__ == "__main__":
    main()