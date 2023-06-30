import torch

from gensim.models import Word2Vec

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
    embeddings_matrix = []
    for sentence in tokenized_corpus:
        embeddings = []
        for word in sentence:
            try:
                embeddings += [w2v.wv[word]]
            except TypeError:
                pass
        embeddings = np.row_stack(embeddings) # (seq_len, embedding_dim)
        embeddings_matrix += [embeddings]
    embed_dim = embeddings_matrix[0].shape[1]
    max_seq_len = max([e.shape[0] for e in embeddings_matrix])
    for i in range(len(embeddings_matrix)):
        embeddings_matrix[i] = np.row_stack(( # pad until max_seq_len is reached
            embeddings_matrix[i],
            np.zeros((max_seq_len-embeddings_matrix[i].shape[0], embed_dim))
        ))
    embeddings_matrix = np.stack(
        embeddings_matrix,
        axis=0
    )
    return torch.from_numpy(embeddings_matrix)