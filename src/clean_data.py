from dataset.twitter_dataset import TwitterDataModule
from preprocessing.tokenize import Tokenizer

def main():
    positive = TwitterDataModule._load_tweets(TwitterDataModule, "twitter-datasets/train_pos.txt", stage="fit")
    negative = TwitterDataModule._load_tweets(TwitterDataModule, "twitter-datasets/train_neg.txt", stage="fit")
    t = Tokenizer(save_to_file="twitter-datasets/train_tokenized.txt")
    t(positive + negative)

    test = TwitterDataModule._load_tweets(TwitterDataModule, "twitter-datasets/test_data.txt", stage="predict")
    t = Tokenizer(save_to_file="twitter-datasets/test_tokenized.txt")
    t(test)

if __name__=="__main__":
    main()