import pytorch_lightning as L

import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
from typing import Callable, Dict

from scipy.sparse._csr import csr_matrix

NEGATIVE = 0
POSITIVE = 1

class TwitterDataModule(L.LightningDataModule):
    def __init__(
        self,
        path_train_pos: str,
        path_train_neg: str,
        path_predict: str,
        convert_to_features: Callable,
        convert_to_features_kwargs: Dict=None,
        tokenizer: Callable=None,
        tokenizer_kwargs: Dict=None,
        val_percentage: float=0.1,
        batch_size: int=32,
    ) -> None:
        super().__init__()
        self.path_train_pos = path_train_pos
        self.path_train_neg = path_train_neg
        self.path_predict = path_predict
        self.val_percentage = val_percentage
        self.convert_to_features = convert_to_features
        self.convert_to_features_kwargs = convert_to_features_kwargs or {}
        self.tokenizer = tokenizer
        self.tokenizer_kwargs = tokenizer_kwargs or {}
        self.batch_size = batch_size

    @property
    def train_corpus(self):
        positive = self._load_tweets(self.path_train_pos)
        negative = self._load_tweets(self.path_train_neg)
        return np.array(positive + negative)

    @property
    def predict_corpus(self):
        return np.array(self._load_tweets(self.path_predict))

    def setup(self, stage: str=None) -> None:
        """Recovers data from disk and performs train/val split"""
        if stage is None or stage == "fit":
            positive = self._load_tweets(self.path_train_pos)[:10]
            negative = self._load_tweets(self.path_train_neg)[:10]
            tweets = np.array(positive + negative)
            if self.tokenizer is not None:
                tweets = self.tokenizer(tweets, **self.tokenizer_kwargs)
            tweets = self.convert_to_features(tweets, **self.convert_to_features_kwargs) 
            if isinstance(tweets, csr_matrix): # CountVectorizer
                tweets = torch.from_numpy(tweets.todense())
            # else: tweets: torch.tensor

            labels = torch.tensor([POSITIVE] * len(positive) + [NEGATIVE] * len(negative)).unsqueeze(1)

            # train, val split
            np.random.seed(1) # reproducibility
            shuffled_indices = np.random.permutation(tweets.shape[0])
            split = int((1 - self.val_percentage) * tweets.shape[0])
            train_indices = shuffled_indices[:split]
            val_indices = shuffled_indices[split:]

            self.train_data = _Dataset(tweets[train_indices], labels[train_indices])
            self.val_data =  _Dataset(tweets[val_indices], labels[val_indices])
            
        if stage is None or stage == "predict":
            test = self.predict_corpus
            tweets = self.convert_to_features(np.array(test))
            if isinstance(tweets, csr_matrix): # CountVectorizer
                tweets = torch.from_numpy(tweets.todense())
            self.test_data = tweets

        self.dims = (self.batch_size, *(tweets.shape[1:])) # save input dimensions
    
    def train_dataloader(self):
        return  DataLoader(self.train_data, self.batch_size)
    
    def val_dataloader(self):
        return DataLoader(self.val_data, self.batch_size)
    
    def predict_dataloader(self):
        return DataLoader(self.test_data, self.batch_size)
    
    def _load_tweets(self, path: str):
        tweets = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                tweets.append(line.rstrip())
        return tweets
    
class _Dataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, i):
        return self.X[i], self.y[i]