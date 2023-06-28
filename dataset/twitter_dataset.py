import pytorch_lightning as L
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

import torch
from torch.utils.data import DataLoader

import numpy as np
from typing import Callable

NEGATIVE = 0
POSITIVE = 1

class TwitterDataModule(L.LightningDataModule):
    def __init__(
        self,
        path_to_train_positive_tweets: str,
        path_to_train_negative_tweets: str,
        path_to_test_tweets: str,
        transform: Callable,
        tokenizer: Callable=None,
        val_percentage: float=0.1,
        batch_size: int=32,
    ) -> None:
        super().__init__()
        self.path_to_train_positive_tweets = path_to_train_positive_tweets
        self.path_to_train_negative_tweets = path_to_train_negative_tweets
        self.path_to_test_tweets = path_to_test_tweets
        self.val_percentage = val_percentage
        self.transform = transform
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def prepare_data(self) -> None:
        """Downloads twitter-dataset and saves data tensors on disk"""
        # train data
        positive = self._load_tweets(self.path_to_train_positive_tweets)
        negative = self._load_tweets(self.path_to_train_negative_tweets)
        # tokenizer
        tweets = self.transform(np.array(positive + negative))
        # if data_augmentation: ...
        labels = torch.tensor([POSITIVE] * len(positive) + [NEGATIVE] * len(negative))
        train_data = {
            'tweets': tweets,
            'labels': labels
        }
        torch.save(train_data, 'twitter_train_data.pt')

        # test data
        test_data = self._load_tweets(self.path_to_test_tweets)
        test_data = self.transform(test_data)
        torch.save(test_data, 'twitter_test_data.pt')

    def setup(self, stage: str=None) -> None:
        """Recovers data from disk and performs train/val split"""
        if stage is None or stage == "fit":
            train_data = torch.load('twitter_train_data.pt')
            tweets, labels = train_data['tweets'], train_data['labels']
            
            # train, val split
            np.random.seed(1) # reproducibility
            shuffled_indices = np.random.permutation(tweets.shape[0])
            split = int((1-self.val_percentage) * tweets.shape[0])
            train_indices = shuffled_indices[:split]
            val_indices = shuffled_indices[split:]

            self.train_data = tweets[train_indices], labels[train_indices]
            self.val_data = tweets[val_indices], labels[val_indices]

        if stage is None or stage == "test":
            self.test_data = torch.load('twitter_test_data.pt')
    
    def _load_tweets(self, path: str):
        tweets = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                tweets.append(line.rstrip())
        return tweets
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return  DataLoader(self.train_data, self.batch_size)
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_data, self.batch_size)
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_data, self.batch_size)