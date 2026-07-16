import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

class Model(nn.Module):   
    def __init__(self, encoder, config, tokenizer, args, num_labels):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args
        self.num_labels = num_labels
        self.dropout = nn.Dropout(args.dropout_probability)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.criterion = CrossEntropyLoss()

    def forward(self, input_ids=None, labels=None): 
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        if labels is not None:
            loss = self.criterion(logits, labels)
            return loss, logits
        else:
            return logits