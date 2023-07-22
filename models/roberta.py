from torch import nn 
from transformers import AutoModelForSequenceClassification

class RoBERTaClassifier(nn.Module):
    def __init__(
        self,
        pretrained_model_name: str="distilroberta-base",
        hidden_dropout_prob=0.25,
    ):
        super().__init__()
        self.roberta = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name, 
            num_labels=1,
            hidden_dropout_prob=hidden_dropout_prob,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: dict
        input_ids = x["input_ids"]
        attention_mask = x["attention_mask"]
        x = self.roberta(input_ids= input_ids, attention_mask=attention_mask)
        x = self.sigmoid(x.logits) # x.logits: regression scores before sigmoid
        return x