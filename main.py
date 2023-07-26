import pytorch_lightning as L
# from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelSummary, LearningRateMonitor, ModelCheckpoint, EarlyStopping

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

def save_predictions(predictions, file_name):
    with open(file_name, 'w+') as file:
        writer = csv.writer(file)
        writer.writerow(['Id', 'Prediction'])
        for i, pred in predictions:
            writer.writerow([i.item(), pred.item()])

def main():
    L.seed_everything(42, workers=True)
    batch_size = 16

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
    #     # convert_to_features=create_w2v_embeddings,
    #     # convert_to_features_kwargs={
    #     #     "workers": 8,
    #     #     "vector_size": embedding_dim,
    #     #     "min_count": 1,
    #     #     "window": 5,
    #     #     "sample": 1e-3,
    #     # },
    #     tokenizer=Tokenizer(
    #         save_to_file="twitter-datasets/train_tokenized.txt"
    #     ),
    #     batch_size=batch_size,
    # )
    # dm.setup("fit")

    # 3) Pretrained Glove embeddings
    # from preprocessing.tokenize import Tokenizer
    # from preprocessing.embeddings import get_pretrained_glove_embeddings, pad_batch

    # embedding_dim = 200
    # dm = TwitterDataModule(
    #     ["twitter-datasets/train_pos.txt", "twitter-datasets/train_neg.txt"],
    #     "twitter-datasets/test_data.txt",
    #     convert_to_features=get_pretrained_glove_embeddings,
    #     convert_to_features_kwargs={
    #         "model_name": "glove-twitter-" + str(embedding_dim), # possible values: 25, 50, 100, 200
    #     },
    #     tokenizer=Tokenizer(),
    #     collate_fn=pad_batch,
    #     batch_size=batch_size,
    #     num_workers=2,
    # )

    # 4) Pretrained Word2Vec embeddings
    from preprocessing.tokenize import Tokenizer
    from preprocessing.embeddings import get_pretrained_word2vec_embeddings, get_embeddings_per_batch

    dm = TwitterDataModule(
        ["twitter-datasets/train_pos.txt", "twitter-datasets/train_neg.txt"],
        "twitter-datasets/test_data.txt",
        # convert_to_features=get_pretrained_word2vec_embeddings,
        # convert_to_features_kwargs={
        #     "model_name": "word2vec-google-news-300", # or: "word2vec-ruscorpora-300"
        # },
        tokenizer=lambda x: [tweet.split() for tweet in x],
        collate_fn=get_embeddings_per_batch,
        # collate_kwargs={
        #     "model_name": "word2vec-google-news-300", # or: "word2vec-ruscorpora-300"
        # },
        batch_size=batch_size,
        num_workers=2,
    )

    # 5) Bert embeddings
    # PRETRAINED_MODEL_NAME =  'cardiffnlp/twitter-roberta-base-sentiment-latest'# 'distilroberta-base' #'distilbert-base-uncased'
    # from transformers import AutoTokenizer

    # print("prepearing data module...")
    # tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
    # dm = TwitterDataModule(
    #     ["twitter-datasets/train_pos_full.txt", "twitter-datasets/train_neg_full.txt"],
    #     "twitter-datasets/test_data.txt",
    #     tokenizer=tokenizer,
    #     tokenizer_kwargs={
    #         "truncation": True,
    #         "padding": True,
    #     },
    #     batch_size=batch_size,
    #     num_workers=2,
    #     val_percentage=0.1
    # )

    # Run datamodule to check input dimensions
    dm.setup(stage="fit")
    print("data module set up.")

    # 2. Model

    # Baselines
    from models.baseline import CNNBaseline, BiRNNBaseline
    model = CNNBaseline()
    model = BiRNNBaseline()

    # from models.bert import CRNNBert
    # model = CRNNBertModel(pretrained_model_name=PRETRAINED_MODEL_NAME)

    # from models.bert import BertUnpooledClassifier
    # from models.attention import SelfAttention3D
    # model = BertUnpooledClassifier(
    #     PRETRAINED_MODEL_NAME,
    #     classifier=nn.Sequential(
    #         SelfAttention3D(
    #             embed_dim=768, 
    #             q_dim=768,
    #             v_dim=256,
    #             collapse=True
    #         ),
    #         nn.Dropout(p=0.5),
    #         nn.Linear(256, 64),
    #         nn.BatchNorm1d(64),
    #         nn.ReLU(),
    #         nn.Dropout(p=0.5),
    #         nn.Linear(64, 8),
    #         nn.BatchNorm1d(8),
    #         nn.ReLU(),
    #         nn.Dropout(p=0.5),
    #         nn.Linear(8, 2),
    #     )
    # )

    # from models.transformer import TransformerClassifier
    # model = TransformerClassifier(
    #     PRETRAINED_MODEL_NAME,
    #     model_kwargs={
    #         "hidden_dropout_prob": 0.1, # "dropout" (BertConfig); "hidden_dropout_prob" (RobertaConfig)
    #         "attention_probs_dropout_prob": 0.1,
    #         # "ignore_mismatched_sizes": True,
    #     }
    # )

    # from models.classifier import BiRNNBaseline
    # model = BiRNNBaseline(
    #     embed_size=300,
    #     hidden_size=200,
    #     num_layers=2
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


    # 3. Lightning net

    net = SentimentAnalysisNet(
        model,
        label_smoothing=0.1,
        lr=1e-5,
        # sched_step_size=3,
        # sched_gamma=0.2,
    )


    # 4. Train

    trainer = L.Trainer(
        max_epochs=6,
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
    print("finished training")
    trainer.validate(net, dm.val_dataloader())

    # path = 'out/{}'.format(timestamp('%d-%m-%Y-%H:%M:%S'))
    # os.makedirs(path, exist_ok=True)
    # torch.save(model.state_dict(), os.path.join(path, '{}.pt'.format(timestamp('%d-%m-%Y-%H:%M:%S'))))


    # 5. Predict

    dm.setup(stage="predict")
    # net = SentimentAnalysisNet.load_from_checkpoint(
    #     "lightning_logs/version_22136887/checkpoints/epoch=3-step=140628.ckpt",
    #     model,
    #     # lr=2e-5,
    #     # sched_step_size=1, # every half an epoch 
    #     # sched_gamma=0.5,
    # )
    print("start prediction...")
    predictions = trainer.predict(net, dm.predict_dataloader())
    path = 'predictions/{}'.format(timestamp('%d-%m-%Y-%H:%M:%S'))
    os.makedirs(path, exist_ok=True)
    save_predictions(torch.vstack(predictions), os.path.join(path, 'predictions.csv'))
    print("finished!")

if __name__ == "__main__":
    main()