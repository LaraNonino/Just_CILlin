import pytorch_lightning as L
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import torch
import torch.nn as nn

from dataset.twitter_dataset import TwitterDataModule
from recipes.sentiment_analysis import SentimentAnalysisNet

if __name__ == "__main__":

    batch_size = 32
    embedding_dim = 300

    from preprocessing.embeddings import create_w2v_embeddings
    from string import punctuation 
    translator = str.maketrans('','', punctuation)

    dm = TwitterDataModule(
        "twitter-datasets/tokenized.txt",
        "twitter-datasets/test_data.txt",
        # 1. Count vectorizer
        # convert_to_features=count_vectorizer.fit_transform,
        # # tokenizer=Tokenizer("tokenized.txt", return_as_matrix=False),
        # tokenizer=lambda x: [tweet.translate(translator) for tweet in x],
        # 2. Word2Vec 
        convert_to_features=create_w2v_embeddings,
        convert_to_features_kwargs={
            "load_path": "twitter-datasets/w2v_tokenized_300.model",
            "workers": 8,
            "vector_size": embedding_dim,
            "min_count": 1,
            "window": 5,
            "sample": 1e-3,
        },
        tokenizer=lambda x: [tweet.translate(translator).split() for tweet in x],
        # save_embeddings_path="w2v_embeddings_300.pt",
        batch_size=batch_size,
    )

    # Run datamodule to check input dimensions
    dm.setup(stage="fit")
    # print(dm.dims)
    quit()

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
    from preprocessing.tokenize import Tokenizer
    from preprocessing.embeddings import create_w2v_embeddings

    dm = TwitterDataModule(
        "twitter-datasets/train_pos_full.txt",
        "twitter-datasets/train_neg_full.txt",
        "twitter-datasets/test_data.txt",
        convert_to_features=create_w2v_embeddings,
        convert_to_features_kwargs={
            "workers": 8,
            "vector_size": embedding_dim,
            "min_count": 1,
            "window": 5,
            "sample": 1e-3,
        },
        tokenizer=Tokenizer(),
        batch_size=batch_size,
    )

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

    # Run datamodule to check input dimensions
    dm.setup(stage="fit")
    print(dm.dims)

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

    # net = SentimentAnalysisNet(
    #     model,
    #     lr=10e-3,
    # )

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

    # trainer = L.Trainer(
    #     max_epochs=1,
    #     # callbacks=trainer_params["callbacks"],
    #     # logger=wandb_logger,
    # )

    # print("start training...")
    # trainer.fit(model=net, datamodule=dm)
    # print("done!")