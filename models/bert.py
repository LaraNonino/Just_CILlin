import torch
from torch import nn
from transformers import DistilBertModel

from models.classifier import CNNBaseline, BiRNNBaseline

class BertPooledClassifier(nn.Module):
    def __init__(
        self,
        pretrained_model_name: str,
        classifier: nn.Module,
    ):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(pretrained_model_name)
        self.dense = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(0.3)
        self.classifier = classifier
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: dict
        input_ids = x["input_ids"]
        attention_mask = x["attention_mask"]
        x = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = self.dropout(x) # x: (batch_size, seq_len, 768)
        # Pool hidden state (equivalen to BertPooler)
        x = self.dense(x[:, 0]) # apply to first token encoding of every sentence
        x = self.activation(x) # x: (batch_size, 768)

        x = self.classifier(x) # x: (batch_size, 1)
        x = self.sigmoid(x)
        return x
    
class BertUnpooledClassifier(nn.Module):
    def __init__(
        self,
        pretrained_model_name: str,
        classifier: nn.Module,
    ):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(pretrained_model_name)
        self.dense = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(0.3)
        self.classifier = classifier
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: dict
        input_ids = x["input_ids"]
        attention_mask = x["attention_mask"]
        x = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = self.dropout(x) # x: (batch_size, seq_len, 768)
    
        x = self.classifier(x) # x: (batch_size, 1)
        x = self.sigmoid(x)
        return x

class CRNNBertModel(nn.Module):
    def __init__(
        self,
        pretrained_model_name: str="distilbert-base-uncased",
        embed_size_cnn: int=768,
        kernel_sizes: list=[3, 4, 5],
        num_channels: list=[100, 100, 100],
        embed_size_rnn: int=100,
        num_hiddens: int=100,
        num_layers: int=2,
    ):
        super().__init__()
        print("getting model")
        self.bert = DistilBertModel.from_pretrained(pretrained_model_name)
        print("model ok")
        self.cnn = CNN1d(
            embed_size=embed_size_cnn,
            kernel_sizes=kernel_sizes,
            num_channels=num_channels,
            output_size=embed_size_rnn,
        )
        self.cnn.apply(init_weights_cnn)
        self.rnn = BiRNN(
            embed_size=embed_size_rnn, 
            num_hiddens=num_hiddens, 
            num_layers=num_layers
        )
        self.rnn.apply(init_weights_rnn)

        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(num_hiddens, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input_ids = x["input_ids"]
        attention_mask = x["attention_mask"]
        output_bert = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_bert[0]
        hidden_state = self.cnn(hidden_state)
        hidden_state = self.rnn(hidden_state)
        hidden_state = self.dropout(hidden_state)
        output = self.classifier(hidden_state)
        x = self.sigmoid(output)
        return x

class BiRNN(nn.Module):
    def __init__(self, embed_size, num_hiddens, num_layers):
        super().__init__()
        # Set `bidirectional` to True to get a bidirectional RNN
        self.encoder = nn.LSTM(embed_size, num_hiddens, num_layers=num_layers, bidirectional=True)
        self.decoder = nn.Linear(4 * num_hiddens, 100)

    def forward(self, inputs):
        inputs = torch.permute(inputs, (1, 0, 2))
        self.encoder.flatten_parameters()
        # Returns hidden states of the last hidden layer at different time
        # steps. The shape of `outputs` is (no. of time steps, batch size,
        # 2 * no. of hidden units)
        outputs, _ = self.encoder(inputs)
        # Concatenate the hidden states at the initial and final time steps as
        # the input of the fully connected layer. Its shape is (batch size,
        # 4 * no. of hidden units)
        encoding = torch.cat((outputs[0], outputs[-1]), dim=1)
        outs = self.decoder(encoding)
        outs = torch.permute(outs, (0, 1))
        return outs

class CNN1d(nn.Module):
    def __init__(self, embed_size, kernel_sizes, num_channels, output_size):
        super().__init__()

        # Create multiple one-dimensional convolutional layers
        self.convs = torch.nn.ModuleList()
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(torch.nn.Conv1d(embed_size, c, k, padding='same'))

        # The max-over-time pooling layer has no parameters, so this instance
        # can be shared
        # self.pool = torch.nn.AdaptiveAvgPool1d(1)
        self.relu = torch.nn.ReLU()

        self.dropout = torch.nn.Dropout(0.5)
        self.decoder = torch.nn.Linear(sum(num_channels), output_size)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)

        encoding = torch.cat([
            torch.squeeze(self.relu(conv(inputs)), dim=-1)
            for conv in self.convs], dim=1)
        
        # print(encoding.shape)
        encoding = encoding.permute(0, 2, 1)

        outputs = self.decoder(self.dropout(encoding))
        return outputs

def init_weights_rnn(module):
    if type(module) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(module.weight)
    if type(module) == torch.nn.LSTM:
        for param in module._flat_weights_names:
            if "weight" in param:
                torch.nn.init.xavier_uniform_(module._parameters[param])

def init_weights_cnn(module):
    if type(module) in (torch.nn.Linear, torch.nn.Conv1d):
        torch.nn.init.xavier_uniform_(module.weight)