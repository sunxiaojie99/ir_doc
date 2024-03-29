import os
import torch
from transformers import BertTokenizer, BertModel, BertConfig
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))

class Cross_Train_Model(nn.Module):
    def __init__(self, h_params, is_predict=False):
        """is_predict: used to control dropout"""
        super().__init__()
        self.pretrained_model_path = h_params.pretrained_model_path
        self.model_config = BertConfig.from_pretrained(
            self.pretrained_model_path)
        self.model_config.output_hidden_states = True
        self.model_config.output_attentions = False
        self.batch_size = h_params.batch_size
        self.is_predict = is_predict

        self.bert_model = BertModel.from_pretrained(
            self.pretrained_model_path, config=self.model_config)
        
        self.attention = nn.Sequential(
            nn.Linear(h_params.hidden_size, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        )

        self.drop = nn.Dropout(p=0.1)

        self.fc = nn.Linear(h_params.hidden_size, h_params.num_labels)

        for param in self.bert_model.parameters():
            param.requires_grad = True
    
    def forward(self, token_ids, token_type_ids, attention_mask):
        outputs = self.bert_model(token_ids, token_type_ids,
                                  attention_mask)  # sequence_output, pooled_output, (hidden_states), (attentions)
        cls_feats = outputs[1]  # [bs, emb]

        sequence_ouput = outputs[0]
        last_layer_hidden_states = outputs.hidden_states[-1]
        weights = self.attention(last_layer_hidden_states)  # (batch_size, sequence_length, 1)

        #   (batch_size, sequence_length, 1) * (batch_size, sequence_length, embedding_size)
        #   (batch_size, embedding_size)
        context_vector = torch.sum(weights * last_layer_hidden_states, dim=1)

        logits = self.fc(context_vector)  # [bs, 2]

        # cls_feats = self.drop(cls_feats)
        # logits = self.fc(cls_feats)  # [bs, 2]

        return logits


