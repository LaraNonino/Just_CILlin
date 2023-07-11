import torch
import pytorch_lightning as L
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, AutoTokenizer
from tqdm import tqdm
import numpy as np

NEGATIVE = 0
POSITIVE = 1

class TWBertDataModule(L.LightningDataModule):
    def __init__(self, path_train_pos: str=None, path_train_neg: str=None, path_predict: str=None, val_percentage: float=0.1, batch_size: int=32, num_workers: int=1):
        super().__init__()
        self.path_train_pos = path_train_pos
        self.path_train_neg = path_train_neg
        self.path_predict = path_predict
        self.val_percentage = val_percentage
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str=None) -> None:
        """Recovers data from disk and performs train/val split"""
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

        if (self.path_train_pos is not None and self.path_train_neg) and (stage is None or stage == "fit"):
            positive = self._load_tweets(self.path_train_pos)
            negative = self._load_tweets(self.path_train_neg)

            tweets = list(positive) + list(negative)
            labels = torch.tensor([POSITIVE] * len(positive) + [NEGATIVE] * len(negative), dtype=torch.float)

            train_data, val_data = self._split_dataset(tweets, labels, self.val_percentage)
            
            econded_train = self.tokenizer(train_data[0], truncation=True, padding=True)
            encoded_val = self.tokenizer(val_data[0], truncation=True, padding=True)

            self.train_dataset = TweetDataset(econded_train, train_data[1])
            self.val_dataset = TweetDataset(encoded_val, val_data[1])
            
        if self.path_predict is not None and stage == "predict":
            test_tweets = self._load_tweets(self.path_predict, stage)
            ids = list(range(1, len(test_tweets)+1))

            encoded_test = self.tokenizer(test_tweets, truncation=True, padding=True)
            self.test_dataset = TweetDataset(encoded_test, ids)
    
    def train_dataloader(self):
        return  DataLoader(self.train_dataset, self.batch_size, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, self.batch_size, num_workers=self.num_workers)
    
    def predict_dataloader(self):
        return DataLoader(self.test_dataset, self.batch_size, num_workers=self.num_workers)
    
    def _load_tweets(self, path: str, stage='fit'):
        tweets = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                tw = line.rstrip()
                if stage == 'predict': tw = tw.split(',', 1)[1]

                tweets.append(tw)
                    
        return tweets
    
    def _split_dataset(self, tweets, labels, val_percentage):
        np.random.seed(42)

        shuffled_indices = np.random.permutation(len(tweets))
        split = int((1 - val_percentage) * len(tweets))

        train_indices = shuffled_indices[:split]
        val_indices = shuffled_indices[split:]

        train_tweets = np.array(tweets)[train_indices].tolist()
        train_labels = np.array(labels)[train_indices].tolist()

        val_tweets = np.array(tweets)[val_indices].tolist()
        val_labels = np.array(labels)[val_indices].tolist()

        return (train_tweets, train_labels), (val_tweets, val_labels)

class TweetDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return item
