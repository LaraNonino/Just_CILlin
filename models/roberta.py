from torch import nn 
from transformers import AutoModelForSequenceClassification

class RoBERTaClassifier(nn.Module):
    def __init__(
        self,
        pretrained_model_name: str="distilroberta-base",
    ):
        super().__init__()
        self.roberta = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name, 
            num_labels=1
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: dict
        input_ids = x["input_ids"]
        attention_mask = x["attention_mask"]
        x = self.roberta(input_ids= input_ids, attention_mask=attention_mask)
        x = self.sigmoid(x.logits)
        return x