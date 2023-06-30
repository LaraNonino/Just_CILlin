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
        path_train_pos: str,
        path_train_neg: str,
        path_predict: str,
        convert_to_features: Callable,
        tokenizer: Callable=None,
        val_percentage: float=0.1,
        batch_size: int=32,
    ) -> None:
        super().__init__()
        self.path_train_pos = path_train_pos
        self.path_train_neg = path_train_neg
        self.path_predict = path_predict
        self.val_percentage = val_percentage
        self.convert_to_features = convert_to_features
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def setup(self, stage: str=None) -> None:
        """Recovers data from disk and performs train/val split"""
        if stage is None or stage == "fit":
            # train_data = torch.load('twitter_train_data.pt')
            # tweets, labels = train_data['tweets'], train_data['labels']
            
            positive = self._load_tweets(self.path_train_pos)
            negative = self._load_tweets(self.path_train_neg)
            tweets = np.array(positive + negative)
            # tokenizer
            tweets = self.convert_to_features(tweets)
            if type(tweets) is not torch.Tensor: # CountVectorizer
                tweets = torch.from_numpy(tweets.todense())
            # if data_augmentation: ...
            labels = torch.tensor([POSITIVE] * len(positive) + [NEGATIVE] * len(negative)).unsqueeze(1)

            # train, val split
            np.random.seed(1) # reproducibility
            shuffled_indices = np.random.permutation(tweets.shape[0])
            split = int((1 - self.val_percentage) * tweets.shape[0])
            train_indices = shuffled_indices[:split]
            val_indices = shuffled_indices[split:]

            self.train_data = torch.hstack((tweets[train_indices], labels[train_indices]))
            self.val_data = torch.hstack((tweets[val_indices], labels[val_indices]))
            
        if stage is None or stage == "predict":
            test = self._load_tweets(self.path_predict)
            tweets = self.convert_to_features(np.array(test))
            if type(tweets) is not torch.Tensor:
                tweets = torch.from_numpy(tweets.todense())
            self.test_data = tweets
    
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