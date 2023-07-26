import torch
from torch.nn.utils.rnn import pad_sequence

from gensim.models import Word2Vec
import gensim.downloader as api

import numpy as np
from tqdm import tqdm
from typing import Callable

def create_w2v_embeddings(tokenized_corpus, **word2vec_kwargs):
    # 1. Get pretrained Word2Vec model or train model
    load_path = word2vec_kwargs.pop("load_path", None) # # retrieve path if present
    if load_path:
        w2v = Word2Vec.load(load_path)
    else:
        save_path = word2vec_kwargs.pop("save_path", None)
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
            except TypeError: # word not in dictionary 
                sim_score, sim_word = w2v.wv.most_similar(word, topn=1)[0]
                if sim_score >= 0.95:
                    embeddings += [w2v.wv[sim_word]]
        if len(embeddings) == 0:
            # print(f"No words for {sentence}!", file=sys.stderr)
            embeddings = [np.zeros((1, word2vec_kwargs["vector_size"]))]
        # embeddings = torch.from_numpy(np.array(embeddings)) # embeddings: (seq_len, embedding_dim)
        embeddings = np.array(embeddings, dtype=object)
        X += [embeddings] # list of torch tensors
    # X = pad_sequence(X, batch_first=True) # (corpus_length, max_seq_len, embedding_dim)
    return X # np.array(X)

def get_pretrained_embeddings(batch, embeddings_model): # e.g. embeddings_model = api.load(model_name)
    X = []
    Y = []
    embedding_dim = embeddings_model.vector_size
    for sentence, y in batch:
        embeddings = []
        for word in sentence:
            if embeddings_model.has_index_for(word):
                embeddings += [embeddings_model.get_vector(word)]
            else:
                embeddings += [np.zeros(embedding_dim)] # unknown token
        embeddings = torch.from_numpy(np.array(embeddings)) # embeddings: (seq_len, embedding_dim)
        X += [embeddings]
        Y += [y]
    X = pad_sequence(X, batch_first=True).float() # (batch_size, max_seq_len, embedding_dim)
    Y = torch.stack(Y).long()
    return X, Y