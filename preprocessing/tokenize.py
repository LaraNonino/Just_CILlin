from typing import Any
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation 

class Tokenizer:
    def __init__(
        self,
        tokenize_kwargs=None,
    ):
        self.tokenize_kwargs = tokenize_kwargs or {}

    def __call__(self, corpus):
        tokenized = []
        for sentence in corpus:
            tokenized += [self.tokenize(sentence, **self.tokenize_kwargs)]
        return tokenized
    
    def tokenize(
        self,
        sentence: str,  
        remove_punctuation: bool=True, 
        remove_stopwords: bool=True, 
        remove_digits: bool=True,
        stem: bool=True
    ):    
        translator = str.maketrans('','', punctuation)
        stoplist = set(stopwords.words('english'))
        stemmer = SnowballStemmer('english')

        sentence = sentence.lower()
        if remove_punctuation:
            sentence = sentence.translate(translator) # remove punctuation
        sentence = sentence.split() # split into tokens
        if remove_stopwords:
            sentence = [w for w in sentence if w not in stoplist] # remove stopwords
        if remove_digits:
            sentence = [w for w in sentence if not w.isdigit()] # normalize numbers
        if stem:
            sentence = [stemmer.stem(w) for w in sentence] # stem each word
        return sentence