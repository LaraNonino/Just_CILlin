import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation 
from ekphrasis.classes.segmenter import Segmenter

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
                f.writelines([" ".join(sentence) + "\n" for sentence in tokenized])
        if not self.return_as_matrix:
            tokenized = [" ".join(sentence) for sentence in tokenized]
        return tokenized
    
    def tokenize(
        self,
        sentence: str,  
        remove_punctuation: bool=False, 
        segment_hashtags: bool=True,
        remove_stopwords: bool=True, 
        remove_digits: bool=True,
        stem: bool=False,
    ):    
        sentence = sentence.lower() 
        if remove_punctuation:
            if segment_hashtags: punctuation.replace("#", "") # keep hashtag
            translator = str.maketrans('','', punctuation)
            sentence = sentence.translate(translator) # remove punctuation
        sentence = sentence.split() # split into tokens
        if segment_hashtags:
            segmenter = Segmenter(corpus="twitter")
            s = []
            for w in sentence:
                if w.startswith("#"):
                    print(w)
                    s += segmenter.segment(w).split()[1:] # remove hashtag
                else:
                    s += [w]
            sentence = s
        if remove_stopwords:
            stoplist = set(stopwords.words('english'))
            s = [w for w in sentence if w not in stoplist] # remove stopwords
            if len(s) > 0:
                sentence = s
        if remove_digits:
            sentence = [w for w in sentence if not w.isdigit()] # normalize numbers
        if stem:
            stemmer = SnowballStemmer('english')
            sentence = [stemmer.stem(w) for w in sentence] # stem each word
        return sentence