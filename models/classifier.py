import torch.nn as nn

class RNNClassifier(nn.Module):
    def __init__(
        self, 
        rnn: nn.Module, # LSTM, GRU
        classifier: nn.Module,
    ):
        super().__init__()
        self.rnn = rnn
        self.classifier = classifier
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch_size, max_seq_len, embedding_size)
        x, _ = self.rnn(x) #  x: (batch_size, max_seq_len, hidden_size)
        x = x[:, -1, :] # only take last hidden state per sentence, encoding whole sequence
        x = self.classifier(x) #  x: (batch_size, 2)
        x = self.sigmoid(x)
        return x