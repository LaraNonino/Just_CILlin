import pytorch_lightning as L
from pytorch_lightning.loggers import WandbLogger
# from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks import ModelSummary

import torch
import torch.nn as nn

import os
from datetime import datetime

from dataset.twitter_dataset import TwitterDataModule
from recipes.sentiment_analysis import SentimentAnalysisNet

def main():
    L.seed_everything(1, workers=True)

    batch_size = 64
    learning_rate = 2e-5

    # from preprocessing.embeddings import create_w2v_embeddings, pad_batch
    # from string import punctuation 
    # translator = str.maketrans('','', punctuation)

    # print("Preparing datamodule")
    # dm = TwitterDataModule(
    #     "twitter-datasets/tokenized.txt",
    #     "twitter-datasets/test_data.txt",
    #     # 1. Count vectorizer
    #     # convert_to_features=count_vectorizer.fit_transform,
    #     # # tokenizer=Tokenizer("tokenized.txt", return_as_matrix=False),
    #     # tokenizer=lambda x: [tweet.translate(translator) for tweet in x],
    #     # 2. Word2Vec 
    #     convert_to_features=create_w2v_embeddings,
    #     convert_to_features_kwargs={
    #         "load_path": "trained_models/w2v_euler_100.model",
    #         "workers": 8,
    #         "vector_size": 100,
    #         "min_count": 1,
    #         "window": 5,
    #         "negative": 0
    #     },
    #     tokenizer=lambda x: [tweet.translate(translator).split() for tweet in x],
    #     collate_fn=pad_batch,
    #     batch_size=batch_size,
    # )

    # # Run datamodule to check input dimensions
    # dm.setup(stage="fit")
    # quit()

    # Choose tokenizer/embedding

    # 1) CountVectorizer:
    # from sklearn.feature_extraction.text import CountVectorizer
    # import nltk
    # nltk.download('stopwords', quiet=True)

    # from nltk.corpus import stopwords
    # count_vectorizer = CountVectorizer(
    #     stop_words=stopwords.words('english'),
    #     max_features=5000 # top features ordered by term frequency across the corpus
    # )
    # dm = TwitterDataModule(
    #     "twitter-datasets/train_pos_full.txt",
    #     "twitter-datasets/train_neg_full.txt",
    #     "twitter-datasets/test_data.txt",
    #     count_vectorizer.fit_transform,
    #     batch_size=32
    # )

    # 2) Word2Vec
    # from preprocessing.tokenize import Tokenizer
    # from preprocessing.embeddings import create_w2v_embeddings

    # dm = TwitterDataModule(
    #     "twitter-datasets/train_pos_full.txt",
    #     "twitter-datasets/train_neg_full.txt",
    #     "twitter-datasets/test_data.txt",
    #     convert_to_features=create_w2v_embeddings,
    #     convert_to_features_kwargs={
    #         "workers": 8,
    #         "vector_size": embedding_dim,
    #         "min_count": 1,
    #         "window": 5,
    #         "sample": 1e-3,
    #     },
    #     tokenizer=Tokenizer(),
    #     batch_size=batch_size,
    # )

    # 3) Pretrained Glove embeddings
    # from preprocessing.tokenize import Tokenizer
    # from preprocessing.embeddings import get_pretrained_glove_embeddings

    # dm = TwitterDataModule(
    #     "twitter-datasets/train_pos_full.txt",
    #     "twitter-datasets/train_neg_full.txt",
    #     "twitter-datasets/test_data.txt",
    #     convert_to_features=get_pretrained_glove_embeddings,
    #     convert_to_features_kwargs={
    #         "dim_name": "glove-twitter-" + str(embedding_dim), # possible values: 25, 50, 100, 200
    #     },
    #     tokenizer=Tokenizer(),
    #     batch_size=batch_size,
    # )

    # 4) Bert embeddings
    PRETRAINED_MODEL_NAME = 'distilbert-base-uncased'
    from transformers import AutoTokenizer

    print("Prepearing data module...")
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
    dm = TwitterDataModule(
        ["twitter-datasets/train_pos_full.txt", "twitter-datasets/train_neg_full.txt"],
        "twitter-datasets/test_data.txt",
        tokenizer=tokenizer,
        tokenizer_kwargs={
            "truncation": True,
            "padding": True,
        },
        batch_size=batch_size,
    )

    # Run datamodule to check input dimensions
    dm.setup(stage="fit")
    print("Data module set up.")

    # Model
    from models.bert import BertPooledClassifier
    from models.attention import SelfAttention
    model = BertPooledClassifier(
        PRETRAINED_MODEL_NAME,
        classifier=nn.Sequential(
            SelfAttention(
                embed_dim=768, 
                q_dim=768,
                v_dim=256,
            ),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )
    )
     
    # from models.bert import BertUnpooledClassifier
    # from models.attention import SelfAttention
    # model = BertUnpooledClassifier(
    #     PRETRAINED_MODEL_NAME,
    #     classifier=nn.Sequential(
    #         SelfAttention(
    #             embed_dim=768, 
    #             q_dim=768,
    #             v_dim=256,
    #             collapse=True
    #         ),
    #         nn.Linear(256, 64),
    #         nn.BatchNorm1d(64),
    #         nn.ReLU(),
    #         nn.Linear(64, 8),
    #         nn.BatchNorm1d(8),
    #         nn.ReLU(),
    #         nn.Linear(8, 1),
    #     )
    # )


    # from models.rnn import RNNClassifier
    # model = RNNClassifier(
    #     rnn=nn.LSTM(
    #         input_size=dm.dims[-1],
    #         hidden_size=64,
    #         num_layers=2,
    #         batch_first=True,
    #         dropout=0.1,
    #     ),
    #     classifier=nn.Sequential(
    #         nn.Linear(64, 16),
    #         nn.BatchNorm1d(16),
    #         nn.ReLU(),
    #         nn.Linear(16, 8),
    #         nn.BatchNorm1d(8),
    #         nn.ReLU(),
    #         nn.Linear(8, 2),
    #         nn.Sigmoid(),
    #     )
    # )
    # print(model)

    net = SentimentAnalysisNet(
        model,
        lr=learning_rate,
    )

    # wandb_logger = WandbLogger(project="cil")
    # lr_monitor = LearningRateMonitor(logging_interval="step")
    # checkpoint_callback = ModelCheckpoint(
    #     monitor='valid/loss',
    #     dirpath='wandb/ckp',
    #     filename='models-{epoch:02d}-{valid_loss:.2f}',
    #     save_top_k=3,
    #     mode='min'
    # )
    # trainer_params = {"callbacks": [lr_monitor, checkpoint_callback]}

    trainer = L.Trainer(
        max_epochs=1,
        # callbacks=trainer_params["callbacks"],
        # logger=wandb_logger,
        callbacks=[ModelSummary(max_depth=1)],
        deterministic=True, 
        log_every_n_steps=125,
        accelerator="gpu",
    )

    print("start training...")
    trainer.fit(model=net, datamodule=dm)
    trainer.validate(net, dm.val_dataloader())

    path = 'out/{}'.format(timestamp('%d-%m-%Y-%H:%M:%S'))
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path, '{}.pt'.format(timestamp('%d-%m-%Y-%H:%M:%S'))))

def timestamp(format):
    ts = datetime.timestamp(datetime.now())
    return datetime.fromtimestamp(ts).strftime(format)

if __name__ == "__main__":
    main()