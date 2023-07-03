import pytorch_lightning as L
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import torch

from dataset.twitter_dataset import TwitterDataModule
from recipes.sentiment_analysis import SentimentAnalysisNet

if __name__ == "__main__":

    batch_size = 32
    embedding_dim = 300

    # 1) CountVectorizer:
    from sklearn.feature_extraction.text import CountVectorizer
    import nltk
    nltk.download('stopwords', quiet=True)

    from nltk.corpus import stopwords
    count_vectorizer = CountVectorizer(
        stop_words=stopwords.words('english'),
        max_features=5000 # top features ordered by term frequency across the corpus
    )
    dm = TwitterDataModule(
        "twitter-datasets/train_pos_full.txt",
        "twitter-datasets/train_neg_full.txt",
        "twitter-datasets/test_data.txt",
        count_vectorizer.fit_transform,
        batch_size=32
    )

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
    #     batch_size=32
    # )

    model = torch.nn.Linear(174, 2)
    net = SentimentAnalysisNet(
        model,
        lr=10e-3,
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

    # trainer = L.Trainer(
    #     max_epochs=2,
    #     callbacks=trainer_params["callbacks"],
    #     logger=wandb_logger,
    # )
    # trainer.fit(model=net, datamodule=dm)