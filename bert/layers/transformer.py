import torch.nn as nn
from bert.layers.layer_norm import LayerNorm
from bert.layers.position_ffn import PositionwiseFeedForward
from bert.layers.multi_headed_attn import MultiHeadAttention

class TransformerLayer(nn.Module):
    """
    Transformer layer mainly consits of two parts:
    multi-head self attention and feed forward layer.
    """
    def __init__(self,args):
        super(TransformerLayer, self).__init__()

        #Multi-head self-attention
        self.self_attn = MultiHeadAttention(
            args.hidden_size,args.head_num,args.dropout
        )
        self.dropout_1 = nn.Dropout(args.dropout)
        self.layer_norm_1 = LayerNorm(args.hidden_size)
        #Feed forward layer.
        self.feed_forward = PositionwiseFeedForward(
            args.hidden_size,args.feedforward_size,args.hidden_size
        )
        self.dropout_2 = nn.Dropout(args.dropout)
        self.layer_norm_2 = LayerNorm(args.hidden_size)
    def forward(self,hidden,mask):
        """"
        Args:
            hidden:[batch_size x seq_length x emb_size]
            mask:[batch_size x 1 x seq_length x seq_length]
        Returns:
            output:[batch_size x seq_length x hidden_sie]
        """
        inter = self.dropout_1(self.self_attn(hidden,hidden,hidden,mask))
        inter = self.layer_norm_1(inter + hidden)
        output = self.dropout_2(self.feed_forward(inter))
        output = self.layer_norm_2(output + inter)
        return output
