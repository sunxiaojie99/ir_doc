import os
import torch
from transformers import BertTokenizer, BertModel, BertConfig
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))


class Dual_Train_Model(nn.Module):
    def __init__(self, h_params, is_prediction=False):
        super().__init__()
        self.pretrained_model_path = h_params.pretrained_model_path
        self.model_config = BertConfig.from_pretrained(
            self.pretrained_model_path)
        self.model_config.output_hidden_states = True
        self.model_config.output_attentions = False
        self.is_prediction = is_prediction
        self.batch_size = h_params.batch_size

        self.bert_model_q = BertModel.from_pretrained(
            self.pretrained_model_path, config=self.model_config)
        self.bert_model_para = BertModel.from_pretrained(
            self.pretrained_model_path, config=self.model_config)
        for param in self.bert_model_q.parameters():
            param.requires_grad = True
        for param in self.bert_model_para.parameters():
            param.requires_grad = True

    def forward(self, q_token_ids, q_token_type_ids, q_attention_mask,
                p_pos_token_ids, p_pos_token_type_ids, p_pos_attention_mask,
                p_neg_token_ids, p_neg_token_type_ids, p_neg_attention_mask):

        # sequence_ouptut 每个token的output, [batch_size, seq_length, embedding_size]
        # pooled_output 句子的output, [batch_size, embedding_size]
        # hidden_states： 很多个(batch_size, sequence_length, embedding_size)
        #                 one for the output of the embeddings + one for the output of each layer
        #                 设置 output_hidden_states=True
        # attentions： (batch_size, num_heads, sequence_length, sequence_length)
        #              设置 output_attentions=True

        query_outputs = self.bert_model_q(q_token_ids, q_token_type_ids, q_attention_mask)
        q_cls_feats = query_outputs[1]  # [bs, emb]

        pos_outputs = self.bert_model_para(p_pos_token_ids, p_pos_token_type_ids, p_pos_attention_mask)
        pos_cls_feats = pos_outputs[1]  # [bs, emb]

        neg_outputs = self.bert_model_para(p_neg_token_ids, p_neg_token_type_ids, p_neg_attention_mask)
        neg_cls_feats = neg_outputs[1]  # [bs, emb]

        p_cls_feats = torch.cat((pos_cls_feats, neg_cls_feats), 0)  # [2bs, emb]

        if self.is_prediction:
            # 对于prediction的时候，pos_cls_feats和neg_cls_feats是一样的
            p_cls_feats = p_cls_feats.narrow_copy(0, 0, self.batch_size) # [bs, emb]
            logits = torch.mul(q_cls_feats, p_cls_feats)  # Element-wise [bs, emb] 手动实现内积
            logits = logits.sum(dim=-1) # [bs] 算22向量间的内积
            graph_vars = {
                "logits": logits,  # [bs]
                "q_rep": q_cls_feats,  # [bs, emb]
                "p_rep": p_cls_feats  # [bs, emb]
            }
            return graph_vars

        # 22做内积
        logits = torch.mm(q_cls_feats, p_cls_feats.T)  # 二维矩阵乘法 [bs, 2bs]

        return logits





