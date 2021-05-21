import math
import torch
import torch.nn as nn
class MultiHeadAttention(nn.Module):
    """
    Each head is self-attention operation.
    self-attention refers to https://arxiv.org/pdf.1706.03762.pdf
    """
    def __init__(self,hidden_size,heads_num,dropout):
        super(MultiHeadAttention,self).__init__()
        self.hidden_size = hidden_size
        self.head_num = heads_num
        self.per_head_size = hidden_size // heads_num

        self.linear_layers = nn.ModuleList([
            nn.Linear(hidden_size,hidden_size) for _ in range(3)
        ])
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(hidden_size,hidden_size)
    def forward(self,key,value,query,mask):
        """

        :param key: [batch_size x seq_length x hidden_size]
        :param value:[batch_size x seq_length x hidden_size]
        :param query:[batch_size x seq_length x hidden_size]
        :param mask:[batch_size x 1 x seq_length x seq_length]
        :return: [batch_size x seq_length x hidden_size]
        """
        batch_size, seq_length, hidden_size = key.size()
        heads_num = self.head_num
        per_head_size = self.per_head_size

        def shape(x):
            return x. \
                transpose(1,2). \
                view(batch_size,seq_length,heads_num,per_head_size). \
                transpose(1,2)
        def unshape(x):
            return x. \
                transpose(1,2). \
                contiguous(). \
                view(batch_size,seq_length,hidden_size)

        query, key, value = [l(x). \
                             view(batch_size,-1,heads_num,per_head_size). \
                             transpose(1,2) \
                             for l, x in zip(self.linear_layers,(query, key, value))
                             ]

        scores = torch.matmul(query,key.transpose(-2,-1))
        scores = scores / math.sqrt(float(per_head_size))# [batch_size x seq_length x seq_length]
        scores = scores + mask
        probs = nn.Softmax(dim=-1)(scores)
        probs = self.dropout(probs)
        output = unshape(torch.matmul(probs,value))
        output = self.final_linear(output)

        return output
