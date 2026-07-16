# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss

    
    
class Model(nn.Module):   
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args
        self.dropout = nn.Dropout(args.dropout_probability)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.criterion = BCEWithLogitsLoss()

        
    def forward(self, input_ids=None, labels=None): 
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        if hasattr(self.args, 'model_type') and self.args.model_type == 'bert':
            pooled_output = last_hidden_state[:, 0, :]  # [CLS] token
        else:
            masked_hidden = last_hidden_state * attention_mask.unsqueeze(-1).float()
            pooled_output = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).float()

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output) 
        
        if labels is not None:
            labels = labels.float()
            loss = self.criterion(logits.squeeze(-1), labels)  

            return loss, torch.sigmoid(logits)
        else:

            return torch.sigmoid(logits)
      
        
 
