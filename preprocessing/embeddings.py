import torch
from torchtext.vocab import FastText
from torch.nn.utils.rnn import pad_sequence

from gensim.models import Word2Vec
import gensim.downloader as api

import numpy as np

def create_w2v_embeddings(tokenized_corpus, **word2vec_kwargs):
    # 1. train Word2Vec
    w2v = Word2Vec(
        tokenized_corpus,
        **word2vec_kwargs,
    )
    w2v.init_sims(replace=True) # done training, so delete context vectors
    # w2v.save('w2v-vectors.pkl')

    # 2. compute embeddings matrix
    X = []
    for sentence in tokenized_corpus:
        embeddings = []
        for word in sentence:
            try:
                embeddings += [w2v.wv[word]]
                # embeddings += [torch.from_numpy(embedding)]
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