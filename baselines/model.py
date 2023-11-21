import torch
import torch.nn as nn
import os


class CustomModel(nn.Module):
    def __init__(self, bert, num_classes, dropout_ratio, pad_num):
        super().__init__()
        self.bert = bert
        self.dropout = nn.Dropout(dropout_ratio)
        self.linear = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.pad_num = pad_num

    def get_attn_mask(self, input_ids):
        return torch.where(input_ids == self.pad_num, torch.tensor(0), torch.tensor(1))

    def forward(self, input_ids):
        attn_mask = self.get_attn_mask(input_ids)
        
        outputs = self.bert(input_ids=input_ids, attention_mask=attn_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)

        logits = self.linear(pooled_output)
        return torch.sigmoid(logits)


class CustomModel2(nn.Module):
    def __init__(self, bert, hidden_dim, num_classes, pad_num, dropout_ratio=0.1):
        super().__init__()
        self.bert = bert
        self.dropout = nn.Dropout(dropout_ratio)
        self.linear = nn.Linear(hidden_dim, num_classes)
        self.pad_num = pad_num

    def get_attn_mask(self, input_ids):
        return torch.where(input_ids == self.pad_num, torch.tensor(0), torch.tensor(1))
    
    def forward(self, input_ids):
        attn_mask = self.get_attn_mask(input_ids)
        # input_ids, attn_mask: [batch_size, max_length] [32, 64]

        emb = self.bert(input_ids=input_ids,
                        attention_mask=attn_mask).last_hidden_state
        # emb: [batch_size, max_length, hidden_size] [32, 64, 768]

        emb = emb[:, 0, :]
        # emb: [batch_size, hidden_size] [32, 768]

        out = self.dropout(emb)
        out = self.linear(out)
        return torch.sigmoid(out)