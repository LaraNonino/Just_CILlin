from torch import nn 
from transformers import AutoModelForSequenceClassification, DistilBertModel

from typing import Dict

class TransformerClassifier(nn.Module):
    def __init__(
        self,
        pretrained_model_name: str="distilroberta-base",
        model_kwargs: Dict=None
    ):
        super().__init__()
        self.transformer = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name, 
            num_labels=2,
            **model_kwargs,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: dict
        input_ids = x["input_ids"]
        attention_mask = x["attention_mask"]
        x = self.transformer(input_ids= input_ids, attention_mask=attention_mask)
        x = x.logits # x.logits: classification scores before sigmoid
        return x
    
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

    def forward(self, x):
        # x: dict
        input_ids = x["input_ids"]
        attention_mask = x["attention_mask"]
        x = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = self.dropout(x) # x: (batch_size, seq_len, 768)
        # Pool hidden state (equivalent to BertPooler)
        x = self.dense(x[:, 0]) # apply to first token encoding of every sentence
        x = self.activation(x) # x: (batch_size, 768)

        x = self.classifier(x) # x: (batch_size, 2)
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

    def forward(self, x):
        # x: dict
        input_ids = x["input_ids"]
        attention_mask = x["attention_mask"]
        x = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = self.dropout(x) # x: (batch_size, seq_len, 768)
    
        x = self.classifier(x) # x: (batch_size, 2)
        return x
