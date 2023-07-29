from functools import partial
from typing import Sequence

import gensim.downloader as api
import pytorch_lightning as L
import torch
from absl import app, flags
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, ModelSummary

from dataset.twitter_dataset import TwitterDataModule
from models.baseline import BiRNNBaseline, CNNBaseline
from preprocessing.embeddings import get_pretrained_embeddings
from recipes.sentiment_analysis import SentimentAnalysisNet

FLAGS = flags.FLAGS
flags.DEFINE_float('lr', 10e-3, 'Learning rate.')
flags.DEFINE_integer('n_epochs', 5, 'Number of epochs to train.')
flags.DEFINE_integer('batch_size', 256, 'Batch size.')
flags.DEFINE_integer('n_workers', 2, 'Numer of workers.')
flags.DEFINE_string('embeddings', 'glove-twitter-100', 'Embeddings to use: glove-twitter-100, glove-twitter-200 or word2vec-google-news-300')
flags.DEFINE_string('baseline', 'cnn', 'Baseline to execute: cnn or birnn')

def get_datamodule(embedding_name: str):
    embeddings = api.load(embedding_name) 
    dm = TwitterDataModule(
        ["twitter-datasets/train_pos_full.txt", "twitter-datasets/train_neg_full.txt"],
        tokenizer=lambda x: [tweet.split() for tweet in x],
        collate_fn=partial(get_pretrained_embeddings, embeddings_model=embeddings),
        batch_size=FLAGS.batch_size,
        num_workers=FLAGS.n_workers,
    )
    return dm

def main(argv: Sequence[str]):
    L.seed_everything(42, workers=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))

    # 1. Dataset
    print("Preparing data module...")
    dm = get_datamodule(FLAGS.embeddings)
    dm.setup(stage="fit")
    print("Data module set up.")
    
    # 2. Model
    embed_size = int(FLAGS.embeddings.split('-')[-1])
    if FLAGS.baseline == 'cnn':
        model = CNNBaseline(embed_size=embed_size, kernel_sizes=[3, 4, 5], num_channels=[100, 100, 100])
    elif FLAGS.baseline == 'birnn':
        model = BiRNNBaseline(embed_size=embed_size)
    else:
        raise NotImplementedError()

    net = SentimentAnalysisNet(model, lr=FLAGS.lr)

    # 3. Train
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