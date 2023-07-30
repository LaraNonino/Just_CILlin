import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


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