import torch
from torchtext.vocab import FastText
from torch.nn.utils.rnn import pad_sequence

from gensim.models import Word2Vec
import gensim.downloader as api

import numpy as np
from tqdm import tqdm

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
    # torch.save(X, "trained_models/w2v_embeddings_100.pt")
    # X = pad_sequence(X, batch_first=True) # (corpus_length, max_seq_len, embedding_dim)
    return X # np.array(X)

def get_pretrained_glove_embeddings(tokenized_corpus, **glove_kwargs):
    model_name = glove_kwargs.get("model_name") or "glove-twitter-300"
    glove_embeddings = api.load(model_name)
    X = []
    for sentence in tokenized_corpus:
        embeddings = []
        for word in sentence:
            if glove_embeddings.has_index_for(word):
                embeddings += [glove_embeddings.get_vector(word)]
        embeddings = torch.from_numpy(np.array(embeddings)) # embeddings: (seq_len, embedding_dim)
        X += [embeddings]
    # X = pad_sequence(X, batch_first=True) # (corpus_length, max_seq_len, embedding_dim)
    return X # np.array(X, dtype=object)

def get_pretrained_word2vec_embeddings(tokenized_corpus, **glove_kwargs):
    model_name = glove_kwargs.get("model_name") or "word2vec-google-news-300"
    w2v_embeddings = api.load(model_name)
    X = []
    for sentence in tokenized_corpus:
        embeddings = []
        for word in sentence:
            if w2v_embeddings.has_index_for(word):
                embeddings += [w2v_embeddings.get_vector(word)]
        embeddings = torch.from_numpy(np.array(embeddings)) # embeddings: (seq_len, embedding_dim)
        X += [embeddings]
    # X = pad_sequence(X, batch_first=True) # (corpus_length, max_seq_len, embedding_dim)
    return X # np.array(X, dtype=object)

def get_pretrained_fasttext_embeddings(tokenized_corpus, **fasttext_kwargs):
    glove_embeddings = FastText(language='en', **fasttext_kwargs)
    X = []
    for sentence in tokenized_corpus:
        X += [glove_embeddings.get_vecs_by_tokens(sentence, lower_case_backup=True)]
    # X = pad_sequence(X, batch_first=True) # (corpus_length, max_seq_len, embedding_dim)
    return np.array(X, dtype=object)

def pad_batch(batch):
    X = []
    Y = []
    for x, y in batch:
        X += [x]
        Y += [y]
    X = pad_sequence(X, batch_first=True)
    Y = torch.cat(Y)
    return X, Y # x: (batch_size, max_seq_len, embedding_dim)