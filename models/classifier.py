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

class BiRNNBaseline(RNNClassifier):
    def __init__(
        self, 
        embed_size, 
        hidden_size, 
        num_layers,
    ):
        lstm = nn.LSTM(
            embed_size, 
            hidden_size, 
            num_layers=num_layers, 
            bidirectional=True, 
            batch_first=True
        )
        classifier = nn.Linear(4 * hidden_size)
        super().__init__(lstm, clasifier)
        self.rnn.apply(init_weights)

    def forward(self, x):
        # x: (batch_size, seq_len, embed_size)
        x, _ = self.rnn(x) # x: (batch_size, 2 * hidden_size)
        x = torch.cat((x[0], x[-1]), dim=1) # (batch_size, 4 * hidden_size)
        x = self.classifier(x) # (batch_size, 1)
        x = self.sigmoid(x)
        return x

    def init_weights(module):
        if type(module) == torch.nn.Linear:
            torch.nn.init.xavier_uniform_(module.weight)
        if type(module) == torch.nn.LSTM:
            for param in module._flat_weights_names:
                if "weight" in param:
                    torch.nn.init.xavier_uniform_(module._parameters[param])

class CNNBaseline(nn.Module):
    def __init__(
        self,
    )
        pass
