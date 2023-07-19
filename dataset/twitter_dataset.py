import pytorch_lightning as L

import torch
from torch.utils.data import Dataset, DataLoader
from transformers.tokenization_utils_base import BatchEncoding

import numpy as np
from typing import List, Dict, Callable, Union

from scipy.sparse._csr import csr_matrix

from tqdm import tqdm

NEGATIVE = 0
POSITIVE = 1

class TwitterDataModule(L.LightningDataModule):
    def __init__(
        self,
        path_train: Union[List[str], str],
        path_predict: str,
        convert_to_features: Callable=None,
        convert_to_features_kwargs: Dict=None,
        tokenizer: Callable=None,
        tokenizer_kwargs: Dict=None,
        collate_fn: Callable=None,
        val_percentage: float=0.1,
        batch_size: int=32,
    ) -> None:
        super().__init__()
        self.path_train = path_train
        self.path_predict = path_predict
        self.val_percentage = val_percentage
        self.convert_to_features = convert_to_features
        self.convert_to_features_kwargs = convert_to_features_kwargs or {}
        self.tokenizer = tokenizer
        self.tokenizer_kwargs = tokenizer_kwargs or {}
        self.collate_fn = collate_fn
        self.batch_size = batch_size

    def setup(self, stage: str=None) -> None:
        """Recovers data from disk and performs train/val split"""
        if stage is None or stage == "fit":
            if isinstance(self.path_train, list):
                positive = self._load_tweets(self.path_train[0])[:10]
                negative = self._load_tweets(self.path_train[1])[:10]
                tweets = positive + negative
                labels = torch.tensor([POSITIVE] * len(positive) + [NEGATIVE] * len(negative), dtype=torch.float).unsqueeze(1)
            elif isinstance(self.path_train, str):
                tweets = self._load_tweets(self.path_train) # file of pre-tokenized training data
                labels = torch.tensor([POSITIVE] * (len(tweets) // 2) + [NEGATIVE] * (len(tweets) // 2), dtype=torch.float).unsqueeze(1) # assuming same number of positive and negative
            
            (train_X, train_y), (val_X, val_y) = self._split_dataset(tweets, labels)

            # Tokenization
            if self.tokenizer:
                train_X = self.tokenizer(train_X, **self.tokenizer_kwargs)
                val_X =  self.tokenizer(val_X, **self.tokenizer_kwargs)

            # Feature extraction
            if self.convert_to_features:
                train_X = self.convert_to_features(train_X, **self.convert_to_features_kwargs) 
                val_X = self.convert_to_features(val_X, **self.convert_to_features_kwargs) 

            if isinstance(train_X, csr_matrix) and isinstance(val_X, csr_matrix): # CountVectorizer
                train_X = torch.from_numpy(train_X.todense()).float()
                val_X = torch.from_numpy(val_X.todense()).float()
            # else: tweets: torch.tensor or np.array

            if isinstance(train_X, BatchEncoding) and isinstance(val_X, BatchEncoding): # bert encodings
                self.train_data = _BertDataset(train_X, train_y)
                self.val_data = _BertDataset(val_X, val_y)
            else: 
                self.train_data = _Dataset(train_X, train_y)
                self.val_data =  _Dataset(val_X, val_y)
            
        if stage is None or stage == "predict":
            test = self.predict_corpus
            tweets = self.convert_to_features(np.array(test))
            if isinstance(tweets, csr_matrix): # CountVectorizer
                tweets = torch.from_numpy(tweets.todense())
            self.test_data = tweets
    
    def train_dataloader(self):
        return  DataLoader(self.train_data, self.batch_size, collate_fn=self.collate_fn)
    
    def val_dataloader(self):
        return DataLoader(self.val_data, self.batch_size, collate_fn=self.collate_fn)
    
    def predict_dataloader(self):
        return DataLoader(self.test_data, self.batch_size, collate_fn=self.collate_fn)
    
    def _load_tweets(self, path: str):
        tweets = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                tweets.append(line.rstrip())
        return tweets
    
    def _split_dataset(self, tweets, labels):
        np.random.seed(1)
        shuffled_indices = np.random.permutation(len(tweets))
        split = int((1 - self.val_percentage) * len(tweets))
        train_indices = shuffled_indices[:split]
        val_indices = shuffled_indices[split:]

        train_tweets = np.array(tweets)[train_indices].tolist()
        val_tweets = np.array(tweets)[val_indices].tolist()

        return (train_tweets, labels[train_indices]), (val_tweets, labels[val_indices])

class _Dataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return self.y.shape[0]
    
    def __getitem__(self, i):
        return self.X[i], self.y[i]
    
class _BertDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {key: torch.tensor(val[i]) for key, val in self.encodings.items()}
        label = self.labels[i]
        return item, label