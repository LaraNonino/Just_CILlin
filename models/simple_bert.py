import torch
import pytorch_lightning as L
from torch.nn import functional as F
from transformers import DistilBertModel

class SABertModel(L.LightningModule):
    def __init__(self, lr):
        super().__init__()
        self.lr = lr

        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.pre_classifier = torch.nn.Sequential(torch.nn.Linear(768, 768), torch.nn.ReLU())
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask):
        output_bert = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_bert[0]
        hidden_state = hidden_state[:, 0]

        hidden_state = self.pre_classifier(hidden_state)
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
