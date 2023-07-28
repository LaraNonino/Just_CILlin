import pytorch_lightning as L

import torch
from torch.utils.data import Dataset, DataLoader
from transformers.tokenization_utils_base import BatchEncoding

import numpy as np
from typing import List, Dict, Callable, Union
from collections import defaultdict

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
        num_workers: int=96,
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
        self.num_workers = num_workers

    def setup(self, stage: str=None) -> None:
        """Recovers data from disk and performs train/val split"""
        if stage is None or stage == "fit":
            if isinstance(self.path_train, list): # both files
                positive = self._load_tweets(self.path_train[0], "fit")
                negative = self._load_tweets(self.path_train[1], "fit")
                tweets = positive + negative
                labels = torch.tensor([POSITIVE] * len(positive) + [NEGATIVE] * len(negative), dtype=torch.long)
            elif isinstance(self.path_train, str): # 1 tokenized file
                tweets = self._load_tweets(self.path_train, "fit") # file of pre-tokenized training data
                labels = torch.tensor([POSITIVE] * (len(tweets) // 2) + [NEGATIVE] * (len(tweets) // 2), dtype=torch.long) # assuming same number of positive and negative
            train_X, train_y, val_X, val_y = self._split_dataset(tweets, labels)

            self.train_data = self._prepare_data(train_X, train_y)
            self.val_data = self._prepare_data(val_X, val_y)
            
        if stage is None or stage == "predict":
            predict_X = self._load_tweets(self.path_predict, "predict") # 10000 samples
            self.predict_data = self._prepare_data(predict_X)
    
    def train_dataloader(self):
        return  DataLoader(self.train_data, self.batch_size, collate_fn=self.collate_fn, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val_data, self.batch_size, collate_fn=self.collate_fn, num_workers=self.num_workers)
    
    def predict_dataloader(self):
        return DataLoader(self.predict_data, self.batch_size, collate_fn=self.collate_fn, num_workers=self.num_workers)
    
    def _load_tweets(self, path: str, stage: str):
        tweets = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                tweet = line.rstrip() 
                if stage == "predict": 
                    tweet = ",".join(tweet.split(",")[1:])
                tweets.append(tweet)
        return tweets
    
    def _split_dataset(self, tweets, labels):
        np.random.seed(1)
        shuffled_indices = np.random.permutation(len(tweets))
        split = int((1 - self.val_percentage) * len(tweets))
        train_indices = shuffled_indices[:split]
        val_indices = shuffled_indices[split:]

        train_tweets = np.array(tweets)[train_indices].tolist()
        val_tweets = np.array(tweets)[val_indices].tolist()

        return train_tweets, labels[train_indices], val_tweets, labels[val_indices]
    
    def _prepare_data(self, X, y=None):
        # tokenize
        if self.tokenizer:
            X = self.tokenizer(X, **self.tokenizer_kwargs)
        # extract features
        if self.convert_to_features:
            X = self.convert_to_features(X, **self.convert_to_features_kwargs) 
        if isinstance(X, csr_matrix): # CountVectorizer
            X = torch.from_numpy(X.todense()).float()
        # dataset 
        if isinstance(X, BatchEncoding) : # bert encodings
            if y is not None:
                dataset = _TransformerDataset(X, y)
            else: 
                dataset = _TransformerPredictDataset(X)
        else: # train_X, val_X: torch.tensor or np.array
            if y is not None:
                dataset = _Dataset(X, y)
            else:
                dataset = _PredictDataset(X)
        return dataset

class _Dataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return self.y.shape[0]
    
    def __getitem__(self, i):
        return self.X[i], self.y[i]

class _PredictDataset(Dataset):
    def __init__(self, X):
        self.X = X 

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i]
    
class _TransformerDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {key: torch.tensor(val[i]) for key, val in self.encodings.items()}
        label = self.labels[i]
        item["labels"] = label # add labels to pass to bert model
        return item, label

class _TransformerPredictDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings 

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, i):
        item = {key: torch.tensor(val[i]) for key, val in self.encodings.items()}
        item["id"] = torch.tensor(i+1, dtype=torch.long) # from 1 to 10000
        return item
    

def collate_wrapper_transformer_dataset(batch, collate_fn=None):
    return_labels = isinstance(first, tuple) # train dataset: (x, y)
    first = batch[0][0] if return_labels else batch[0] # predict dataset: x
    X = defaultdict(list)
    for k, v in first.item():
        X[k] = torch.stack([x[k] for x, _ in batch])
        if collate_fn:
            X = collate_fn(X)
    if return_labels:
        Y = torch.stack([y for _, y in batch]).long()
        return X, Y
    return X