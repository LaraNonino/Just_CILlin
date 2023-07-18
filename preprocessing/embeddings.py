import torch
from torchtext.vocab import FastText
from torch.nn.utils.rnn import pad_sequence

from gensim.models import Word2Vec
import gensim.downloader as api

import numpy as np

from tqdm import tqdm

def create_w2v_embeddings(tokenized_corpus, **word2vec_kwargs):
    # 1. Get pretrained Word2Vec model or train model
    load_path = word2vec_kwargs.pop("load_path", None)
    if load_path:
        w2v = Word2Vec.load(load_path)
    else:
        save_path = word2vec_kwargs.pop("save_path", None) # retrieve path if is keywords
        w2v = Word2Vec(
            tokenized_corpus,
            **word2vec_kwargs,
        )
        if save_path:
            w2v.save(save_path)

    # 2. compute embeddings matrix
    X = []
    for sentence in tqdm(tokenized_corpus):
        if isinstance(sentence, str):
            sentence = sentence.split()
        embeddings = []
        for word in sentence:
            try:
                embeddings += [w2v.wv[word]]
            except TypeError:
                pass
        embeddings = torch.from_numpy(np.array(embeddings)) # embeddings: (seq_len, embedding_dim)
        X += [embeddings]
    X = pad_sequence(X, batch_first=True) # (batch_size, max_seq_len, embedding_dim)
    return X

def get_pretrained_glove_embeddings(tokenized_corpus, **glove_kwargs):
    dim_name = glove_kwargs.get("dim_name") or "glove-twitter-300"
    glove_embeddings = api.load(dim_name)
    X = []
    for sentence in tokenized_corpus:
        embeddings = []
        for word in sentence:
            if glove_embeddings.has_index_for(word):
                embeddings += [glove_embeddings.get_vector(word)]
        embeddings = torch.from_numpy(np.array(embeddings)) # embeddings: (seq_len, embedding_dim)
        X += [embeddings]
    X = pad_sequence(X, batch_first=True) # (batch_size, max_seq_len, embedding_dim)
    return X

def get_pretrained_fasttext_embeddings(tokenized_corpus, **fasttext_kwargs):
    glove_embeddings = FastText(language='en', **fasttext_kwargs)
    X = []
    for sentence in tokenized_corpus:
        X += [glove_embeddings.get_vecs_by_tokens(sentence, lower_case_backup=True)]
    X = pad_sequence(X, batch_first=True) # (batch_size, max_seq_len, embedding_dim)
    return X