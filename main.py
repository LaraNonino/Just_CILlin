import pytorch_lightning as L
import torch

from dataset.twitter_dataset import TwitterDataModule
from recipes.sentiment_analysis import SentimentAnalysisNet

from sklearn.feature_extraction.text import CountVectorizer

if __name__ == "__main__":
    batch_size = 32
    embedding_dim = 5000
    dm = TwitterDataModule(
        "twitter-datasets/train_pos_full.txt",
        "twitter-datasets/train_neg_full.txt",
        "twitter-datasets/test_data.txt",
        CountVectorizer(max_features=embedding_dim).fit_transform,
        batch_size=32
    )

    dm.setup(stage="fit")
    print("done")

    model = 

    # model = torch.nn.Linear(174, 2)
    # net = SentimentAnalysisNet(
    #     model,
    #     lr=10e-3,
    # )
    # trainer = L.Trainer(max_epochs=2)
    # trainer.fit(model=net, datamodule=dm)