import torch
import torch.nn as nn

from transformers import AutoModelForSequenceClassification

from typing import Union

from peft import (
    get_peft_model,
    PromptTuningConfig,
    PrefixTuningConfig,
    PromptEncoderConfig,
)

class PTunedlassifier(nn.Module):
    def __init__(
        self, 
        pretrained_model_name: str, 
        peft_config: Union[PromptTuningConfig, PrefixTuningConfig, PromptEncoderConfig],
        classifier: nn.Module=None,
    ):
        super().__init__()
        if classifier:
            model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                pretrained_model_name,
                num_labels=2
            )
        self.model = get_peft_model(model, peft_config) # wrap in PeftModel (disables grad in transformer model)
        print((self.model).print_trainable_parameters())
        self.classifier = classifier

    def forward(self, x):
        # x: dict
        input_ids = x["input_ids"]
        attention_mask = x["attention_mask"]
        x = self.model(input_ids=input_ids, attention_mask=attention_mask)
        if self.classifier:
            x = x.last_hidden_state # x: (batch_size, seq_len, embedding_dim)
            x = self.classifier(x) # x: (batch_size, 2)
        else:
            x = x.logits # x: (batch_size, 2)
        return x 
         