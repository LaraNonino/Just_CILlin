class CNNBase(nn.Module):
    def __init__(
        self,
    )
        pass

class BiRNNBaseline(nn.Module):
    def __init__(
        self, 
        embed_size, 
        num_hiddens, 
        num_layers,
        classifier,
    ):
        super().__init__()
        self.lstm = nn.LSTM(embed_size, num_hiddens, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.lstm.apply(init_weights)
        self.classifier = classifier

    def forward(self, x):
        # x: (batch_size, seq_len, embed_size)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x) # x: (batch_size, 2 * num_hidden)
        x = torch.cat((x[0], x[-1]), dim=1) # (batch_size, 4 * num_hidden)
        x = self.classifier(x) # (batch_size, 1)
        return x

    def init_weights(module):
        if type(module) == torch.nn.Linear:
            torch.nn.init.xavier_uniform_(module.weight)
        if type(module) == torch.nn.LSTM:
            for param in module._flat_weights_names:
                if "weight" in param:
                    torch.nn.init.xavier_uniform_(module._parameters[param])
