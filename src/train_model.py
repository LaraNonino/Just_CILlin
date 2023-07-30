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
flags.DEFINE_float('lr', 2e-5, 'Learning rate.')
flags.DEFINE_integer('n_epochs', 5, 'Number of epochs to train.')
flags.DEFINE_integer('batch_size', 256, 'Batch size.')
flags.DEFINE_integer('n_workers', 2, 'Numer of workers.')
flags.DEFINE_float('label_smoothing', 0, 'Label smoothing.')
flags.DEFINE_integer('sched_step_size', None, 'Learning rate scheduler step size.')
flags.DEFINE_float('sched_gamma', None, 'Learning rate scheduler gamma.')

PRETRAINED_MODEL_NAME = 'distilbert-base-uncased'

def main(argv: Sequence[str]):
    L.seed_everything(42, workers=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))

    # 1. Dataset
    print("Preparing data module...")

    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
    dm = TwitterDataModule(
        ["twitter-datasets/train_pos.txt", "twitter-datasets/train_neg.txt"],
        tokenizer=tokenizer,
        tokenizer_kwargs={"truncation": True, "padding": True},
        batch_size=FLAGS.batch_size,
        num_workers=FLAGS.n_workers,
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
        label_smoothing=FLAGS.label_smoothing,
        lr=FLAGS.lr,
        sched_step_size=FLAGS.sched_step_size,
        sched_gamma=FLAGS.sched_gamma
    )

    # 4. Train
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
    )

    print("Start training...")
    trainer.fit(net, dm.train_dataloader(), dm.val_dataloader())
    print("Finished training")
    trainer.validate(net, dm.val_dataloader())
    
    print("Finished!")

if __name__ == "__main__":
    app.run(main)