from torch import nn 
from transformers import AutoModelForSequenceClassification

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
            num_labels=1,
            **model_kwargs,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: dict
        input_ids = x["input_ids"]
        attention_mask = x["attention_mask"]
        x = self.transformer(input_ids= input_ids, attention_mask=attention_mask)
        x = self.sigmoid(x.logits) # x.logits: regression scores before sigmoid
        return x