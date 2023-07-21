import pytorch_lightning as L
# from pytorch_lightning.loggers import WandbLogger
# from pytorch_lightning.callbacks import ModelSummary, LearningRateMonitor

import torch
import torch.nn as nn

import os
import csv
from datetime import datetime
import math

from dataset.twitter_dataset import TwitterDataModule
from recipes.sentiment_analysis import SentimentAnalysisNet

def timestamp(format):
    ts = datetime.timestamp(datetime.now())
    return datetime.fromtimestamp(ts).strftime(format)

def save_predictions(preds, file_name):
    with open(file_name, 'w+') as file:
        writer = csv.writer(file)
        writer.writerow(['Id', 'Prediction'])
        for i, pred in enumerate(preds):
            pred = -1 if pred.item() == 0 else 1
            writer.writerow([i, pred])

def main():
    L.seed_everything(42, workers=True)

    batch_size = 64

    # 1. Dataset
    
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
    #     ["twitter-datasets/train_pos_full.txt", "twitter-datasets/train_neg_full.txt"],
    #     "twitter-datasets/test_data.txt",
    #     count_vectorizer.fit_transform,
    #     batch_size=32
    # )

    # 2) Word2Vec
    # from preprocessing.tokenize import Tokenizer
    # from preprocessing.embeddings import create_w2v_embeddings

    # dm = TwitterDataModule(
    #     ["twitter-datasets/train_pos_full.txt", "twitter-datasets/train_neg_full.txt"],
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
    #     ["twitter-datasets/train_pos_full.txt", "twitter-datasets/train_neg_full.txt"],
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

    print("prepearing data module...")
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
    print("data module set up.")
    

    # 2. Model

    # from models.bert import BertPooledClassifier
    # from models.attention import SelfAttention
    # model = BertPooledClassifier(
    #     PRETRAINED_MODEL_NAME,
    #     classifier=nn.Sequential(
    #         SelfAttention(
    #             embed_dim=768, 
    #             q_dim=768,
    #             v_dim=256,
    #         ),
    #         nn.BatchNorm1d(256),
    #         nn.ReLU(),
    #         nn.Linear(256, 64),
    #         nn.BatchNorm1d(64),
    #         nn.ReLU(),
    #         nn.Linear(64, 8),
    #         nn.BatchNorm1d(8),
    #         nn.ReLU(),
    #         nn.Linear(8, 1),
    #     )
    # )
     
    from models.bert import BertUnpooledClassifier
    from models.attention import SelfAttention3D
    model = BertUnpooledClassifier(
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
            nn.Linear(8, 1),
        )
    )

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


    # 3. Lightning net

    net = SentimentAnalysisNet(
        model,
        lr=2e-5,
        sched_step_size=1,
        sched_gamma=0.25,
    )


    # 4. Train

    trainer = L.Trainer(
        max_epochs=6,
        # callbacks=[ModelSummary(max_depth=3)], # , LearningRateMonitor(logging_interval='step')],
        deterministic=True, 
        log_every_n_steps=100,
        accelerator="gpu",
    )

    print("start training...")
    trainer.fit(model=net, datamodule=dm)
    print("finished training")
    trainer.validate(net, dm.val_dataloader())

    # path = 'out/{}'.format(timestamp('%d-%m-%Y-%H:%M:%S'))
    # os.makedirs(path, exist_ok=True)
    # torch.save(model.state_dict(), os.path.join(path, '{}.pt'.format(timestamp('%d-%m-%Y-%H:%M:%S'))))


    # 5. Predict

    # dm.setup(stage="predict")
    # net = SentimentAnalysisNet.load_from_checkpoint(
    #     "lightning_logs/version_22136887/checkpoints/epoch=3-step=140628.ckpt",
    #     model,
    #     # lr=2e-5,
    #     # sched_step_size=(2500000*0.9//batch_size) // 2, # every half an epoch 
    #     # sched_gamma=0.5,
    # )
    # print("start prediction...")
    # predictions = trainer.predict(net, dm.predict_dataloader())
    # path = 'predictions/{}'.format(timestamp('%d-%m-%Y-%H:%M:%S'))
    # os.makedirs(path, exist_ok=True)
    # save_predictions(torch.vstack(predictions), os.path.join(path, 'predictions.csv'))

if __name__ == "__main__":
    main()