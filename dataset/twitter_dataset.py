import pytorch_lightning as L

import torch
from torch.utils.data import Dataset, DataLoader

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
        convert_to_features: Callable,
        convert_to_features_kwargs: Dict=None,
        tokenizer: Callable=None,
        tokenizer_kwargs: Dict=None,
        save_embeddings_path: str=None,
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
        self.save_embeddings_path = save_embeddings_path
        self.batch_size = batch_size

    def setup(self, stage: str=None) -> None:
        """Recovers data from disk and performs train/val split"""
        if stage is None or stage == "fit":
            if isinstance(self.path_train, list):
                positive = self._load_tweets(self.path_train[0])
                negative = self._load_tweets(self.path_train[1])
                tweets = np.array(positive + negative)
                labels = torch.tensor([POSITIVE] * len(positive) + [NEGATIVE] * len(negative), dtype=torch.float).unsqueeze(1)
            elif isinstance(self.path_train, str):
                tweets = self._load_tweets(self.path_train) # file of pre-tokenized training data
                labels = torch.tensor([POSITIVE] * (len(tweets) // 2) + [NEGATIVE] * (len(tweets) // 2), dtype=torch.float).unsqueeze(1) # assuming same number of positive and negative
            
            # Tokenization
            if self.tokenizer is not None:
                tweets = self.tokenizer(tweets, **self.tokenizer_kwargs) 
                
            # Feature extraction
            tweets = self.convert_to_features(tweets, **self.convert_to_features_kwargs) 
            if isinstance(tweets, csr_matrix): # CountVectorizer
                tweets = torch.from_numpy(tweets.todense()).float()
            # else: tweets: torch.tensor
            
            # Save embeddings if needed
            if self.save_embeddings_path:
                torch.save(tweets, self.save_embeddings_path)
            
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
            for line in tqdm(f):
                tweets.append(line.rstrip())
        return tweets

class RecoverTwitterEmbeddingsDataModule(L.LightningDataModule):
    def __init__(
        self,
        path_train_embeddings: str,
        path_predict_embeddings: str=None,
        val_percentage: float=0.1,
        batch_size: int=32,
    ) -> None:
        self.path_train_embeddings = path_train_embeddings
        self.path_predict_embeddings = path_predict_embeddings
        self.val_percentage = val_percentage
        self.batch_size = batch_size

        self.corpus_length = 2500000 # length of full training dataset

    def setup(self, stage: str=None) -> None:
        """Recovers embeddings from disk and performs train/val split"""
        if stage is None or stage == "fit":
            tweets = torch.load(self.path_train_embeddings)
            labels = labels = torch.tensor([POSITIVE] * (self.corpus_length // 2) + [NEGATIVE] * (self.corpus_length // 2), dtype=torch.float).unsqueeze(1) # assuming using full dataset
            
            # train, val split
            np.random.seed(1) # reproducibility
            shuffled_indices = np.random.permutation(tweets.shape[0])
            split = int((1 - self.val_percentage) * tweets.shape[0])
            train_indices = shuffled_indices[:split]
            val_indices = shuffled_indices[split:]

            self.train_data = _Dataset(tweets[train_indices], labels[train_indices])
            self.val_data =  _Dataset(tweets[val_indices], labels[val_indices])
        
        if stage is None or stage == "predict":
            tweets = torch.load(self.path_predict_embeddings)
        
        self.dims = (self.batch_size, *(tweets.shape[1:])) 

class _Dataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, i):
        return self.X[i], self.y[i]