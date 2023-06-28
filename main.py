from dataset.twitter_dataset import TwitterDataModule

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

if __name__ == "__main__":
    dm = TwitterDataModule(
        "twitter-datasets/train_pos_full.txt",
        "twitter-datasets/train_neg_full.txt",
        "twitter-datasets/test_data.txt",
        CountVectorizer(max_features=5000).fit_transform,
        batch_size=32
    )
    dm.prepare_data()
    dm.setup()