import torch
import torch.nn as nn

def init_weights(module):
    if type(module) in (nn.Linear, nn.Conv1d):
        nn.init.xavier_uniform_(module.weight)
    if type(module) == nn.LSTM:
        for param in module._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(module._parameters[param])

class RNNClassifier(nn.Module):
    def __init__(
        self, 
        rnn: nn.Module, # LSTM, GRU
        classifier: nn.Module,
    ):
        super().__init__()
        self.rnn = rnn
        self.classifier = classifier

    def forward(self, x):
        # x: (batch_size, max_seq_len, embedding_size)
        x, _ = self.rnn(x) #  x: (batch_size, max_seq_len, hidden_size)
        x = x[:, -1, :] # only take last hidden state per sentence, encoding whole sequence
        x = self.classifier(x) #  x: (batch_size, 2)
        return x

class BiRNNBaseline(RNNClassifier):
    def __init__(
        self, 
        embed_size: int, 
        hidden_size: int=100, 
        num_layers: int=2,
    ):
        lstm = nn.LSTM(
            embed_size, 
            hidden_size, 
            num_layers=num_layers, 
            bidirectional=True, 
            batch_first=True
        )
        classifier = nn.Linear(4 * hidden_size, 2)
        super().__init__(lstm, classifier)
        self.apply(init_weights)

    def forward(self, x):
        # x: (batch_size, seq_len, embed_size)
        x, _ = self.rnn(x) # x: (batch_size, seq_len, 2 * hidden_size)
        x = torch.cat((x[:, 0], x[:, -1]), dim=1) # (batch_size, 4 * hidden_size)
        x = self.classifier(x) # (batch_size, 2)
        return x

class CNNBaseline(nn.Module):
    def __init__(
        self, 
        embed_size: int, 
        kernel_sizes: list, 
        num_channels: list,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(embed_size, c, k))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.relu = nn.ReLU()
        
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(sum(num_channels), 2) 
        
        self.apply(init_weights)

    def forward(self, x):
        # x: (batch_size, seq_len, embed_size)
        x = x.permute(0, 2, 1) # (batch_size, embed_size, seq_len)
        encoding = torch.cat([
            torch.squeeze(self.relu(
                self.pool(
                    conv(x) # x: (batch_size, num_channels[i], seq_len)
                )), # x: (batch_size, num_channels[i], 1)
            dim=-1) for conv in self.convs
        ], dim=1) # encoding: (batch_size, sum(num_channels))
        x = self.classifier(self.dropout(encoding)) # x: (batch_size, 2)
        return x