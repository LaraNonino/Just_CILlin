from dataset.twitter_dataset import TwitterDataModule
from preprocessing.tokenize import Tokenizer

def main():
    positive = TwitterDataModule._load_tweets(TwitterDataModule, "twitter-datasets/train_pos_full.txt", stage="fit")
    negative = TwitterDataModule._load_tweets(TwitterDataModule, "twitter-datasets/train_neg_full.txt", stage="fit")
    t = Tokenizer(save_to_file="train_tokenized.txt")
    t(positive + negative)

    # test = TwitterDataModule._load_tweets(TwitterDataModule, "twitter-datasets/test_data.txt", stage="predict")
    # t = Tokenizer(save_to_file="test_tokenized.txt")
    # t(test)

if __name__=="__main__":
    main()