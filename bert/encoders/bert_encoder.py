import torch.nn as nn
from bert.layers.transformer import TransformerLayer

class BertEncoder(nn.Module):
    """
    BERT encoder exploits 12 or 24 transformer layers to extract feature
    """
    def __init__(self,args):
        super(BertEncoder, self).__init__()
        self.layers_num = args.layers_num
        self.parameter_sharing = args.parateter_sharing
        self.factorized_embedding_parameterization = args.factorized_embedding_parameterization

        if self.factorized_embedding_parameterization:
            self.linear = nn.Linear(args.emb_size,args.hidden_size)

        if self.parameter_sharing:
            self.transformer = TransformerLayer(args)
        else:
            self.transformer = nn.ModuleList([
                TransformerLayera(args) for _ in range(self.layers_num)
            ])