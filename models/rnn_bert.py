import torch
import pytorch_lightning as L
from torch.nn import functional as F
from transformers import DistilBertModel

class RNNBertModel(L.LightningModule):
    def __init__(self, lr):
        super().__init__()
        self.lr = lr

        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        # self.bert.weight.requires_grad = False

        # self.pre_classifier = torch.nn.Sequential(torch.nn.Linear(768, 768), torch.nn.ReLU())
        self.rnn = BiRNN(embed_size=768, num_hiddens=100, num_layers=2)
        self.rnn.apply(init_weights)

        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(100, 2)

    def forward(self, input_ids, attention_mask):
        output_bert = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_bert[0]
        hidden_state = self.rnn(hidden_state)
        hidden_state = self.dropout(hidden_state)
        output = self.classifier(hidden_state)
        return output

    def cross_entropy_loss(self, logits, labels):
        return F.cross_entropy(logits, labels)

    def training_step(self, train_batch, batch_idx):
        ids = train_batch['input_ids']
        mask = train_batch['attention_mask']
        labels = train_batch['labels']

        logits = self.forward(ids, mask)
        loss = self.cross_entropy_loss(logits, labels)

        _, preds = torch.max(logits, dim=1)
        acc = (self._accuracy(preds, labels) * 100) / labels.size(0) 
        
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    def validation_step(self, val_batch, batch_idx):
        ids = val_batch['input_ids']
        mask = val_batch['attention_mask']
        labels = val_batch['labels']
        
        logits = self.forward(ids, mask)
        loss = self.cross_entropy_loss(logits, labels)

        _, preds = torch.max(logits, dim=1)
        acc = (self._accuracy(preds, labels) * 100) / labels.size(0) 
        
        self.log('val_loss', loss)
        self.log('val_acc', acc)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        ids = batch['input_ids']
        mask = batch['attention_mask']
        test_id = batch['labels']

        logits = self.forward(ids, mask)
        _, preds = torch.max(logits, dim=1)

        return torch.stack((test_id, preds), dim=1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def _accuracy(self, prediction, target):
        return (prediction == target).sum().item()


class BiRNN(torch.nn.Module):
    def __init__(self, embed_size, num_hiddens, num_layers, **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        # Set `bidirectional` to True to get a bidirectional RNN
        self.encoder = torch.nn.LSTM(embed_size, num_hiddens, num_layers=num_layers, bidirectional=True)
        self.decoder = torch.nn.Linear(4 * num_hiddens, 100)

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

def init_weights(module):
    if type(module) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(module.weight)
    if type(module) == torch.nn.LSTM:
        for param in module._flat_weights_names:
            if "weight" in param:
                torch.nn.init.xavier_uniform_(module._parameters[param])
