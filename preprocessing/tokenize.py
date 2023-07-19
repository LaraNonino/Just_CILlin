import nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation 

# from transformers import AutoTokenizer

from tqdm import tqdm

class Tokenizer:
    def __init__(
        self,
        save_to_file: str=None,
        return_as_matrix: bool=True,
        tokenize_kwargs=None,
    ):
        self.path = save_to_file
        self.return_as_matrix = return_as_matrix
        self.tokenize_kwargs = tokenize_kwargs or {}

    def __call__(self, corpus):
        tokenized = []
        for sentence in tqdm(corpus):
            tokenized += [self.tokenize(sentence, **self.tokenize_kwargs)]
        if self.path:
            with open(self.path, "w") as f:
                f.writelines([" ".join(sentence)+"\n" for sentence in tokenized])
        if not self.return_as_matrix:
            tokenized = [" ".join(sentence) for sentence in tokenized]
        return tokenized
    
    def tokenize(
        self,
        sentence: str,  
        remove_punctuation: bool=False, 
        remove_stopwords: bool=True, 
        remove_digits: bool=True,
        stem: bool=True,
    ):    
        translator = str.maketrans('','', punctuation)
        stoplist = set(stopwords.words('english'))
        stoplist.add("user") # twitter_dataset
        stemmer = SnowballStemmer('english')

        sentence = sentence.lower()
        if remove_punctuation:
            sentence = sentence.translate(translator) # remove punctuation
        sentence = sentence.split() # split into tokens
        if remove_stopwords:
            s = [w for w in sentence if w not in stoplist] # remove stopwords
            if len(s) > 0:
                sentence = s
        if remove_digits:
            sentence = [w for w in sentence if not w.isdigit()] # normalize numbers
        if stem:
            sentence = [stemmer.stem(w) for w in sentence] # stem each word
        return sentence

# class BertTWTokenizer:
#     def __init__(
#         self,
#         pretrained_model_name: str='bert-base-uncased',
#     )
#         self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)