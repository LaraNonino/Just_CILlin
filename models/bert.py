import torch
from torch import nn
from transformers import DistilBertModel

# from models.baseline import init_weights


# class CRNNBert(nn.Module):
#     def __init__(
#         self,
#         pretrained_model_name: str="distilbert-base-uncased",
#         embed_size_cnn: int=768,
#         kernel_sizes: list=[3, 4, 5],
#         num_channels: list=[100, 100, 100],
#         embed_size_rnn: int=100,
#         num_hiddens: int=100,
#         num_layers: int=2,
#     ):
#         super().__init__()
#         # self.bert = DistilBertModel.from_pretrained(pretrained_model_name)
#         self.cnn = CNN1d(
#             embed_size=embed_size_cnn,
#             kernel_sizes=kernel_sizes,
#             num_channels=num_channels,
#             output_size=embed_size_rnn,
#         )
#         self.cnn.apply(init_weights)
#         self.rnn = BiRNN(
#             embed_size=embed_size_rnn, 
#             num_hiddens=num_hiddens, 
#             num_layers=num_layers
#         )
#         self.rnn.apply(init_weights)

#         self.dropout = nn.Dropout(0.3)
#         self.classifier = nn.Linear(num_hiddens, 2)

#     def forward(self, x):
#         # x: dict
#         # input_ids = x["input_ids"]
#         # attention_mask = x["attention_mask"]
#         # x = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state # last_hidden_state: (batch_size, sequence_length, hidden_size)
#         x = self.cnn(x)
#         x = self.rnn(x)
#         x = self.dropout(x)
#         x = self.classifier(x)
#         return x

# class BiRNN(nn.Module):
#     def __init__(self, embed_size, num_hiddens, num_layers):
#         super().__init__()
#         self.encoder = nn.LSTM(embed_size, num_hiddens, num_layers=num_layers, bidirectional=True)
#         self.decoder = nn.Linear(4 * num_hiddens, 100)

#     def forward(self, x):
#         x = torch.permute(x, (1, 0, 2))
#         print(f"3: {x.shape}")
#         x, _ = self.encoder(x)
#         print(f"4: {x.shape}")
#         x = torch.cat((x[0], x[-1]), dim=1) # initial and final time steps
#         print(f"5: {x.shape}")
#         x = self.decoder(x)
#         print(f"5: {x.shape}")
#         x = torch.permute(x, (0, 1))
#         return x

# class CNN1d(nn.Module):
#     def __init__(
#         self, 
#         embed_size, 
#         kernel_sizes, 
#         num_channels, 
#         output_size
#     ):
#         super().__init__()

#         # Create multiple one-dimensional convolutional layers
#         self.convs = torch.nn.ModuleList()
#         for c, k in zip(num_channels, kernel_sizes):
#             self.convs.append(torch.nn.Conv1d(embed_size, c, k, padding='same'))

#         # The max-over-time pooling layer has no parameters, so this instance
#         # can be shared
#         # self.pool = torch.nn.AdaptiveAvgPool1d(1)
#         self.relu = torch.nn.ReLU()

#         self.dropout = torch.nn.Dropout(0.5)
#         self.decoder = torch.nn.Linear(sum(num_channels), output_size)

#     def forward(self, x):
#         # x: (batch_size, seq_len, embed_size)
#         x = x.permute(0, 2, 1) # x: (batch_size, embed_size, seq_len)
#         encoding = torch.cat([
#             torch.squeeze(self.relu(conv(x)), dim=-1)
#             for conv in self.convs
#         ], dim=1)
#         print(f"1: {encoding.shape}")
#         encoding = encoding.permute(0, 2, 1)

#         outputs = self.decoder(self.dropout(encoding))
#         print(f"2: {outputs.shape}")
#         return outputs