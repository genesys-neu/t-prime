import math
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TransformerModel(nn.Module):

    def __init__(self, d_model: int = 512, nhead: int = 8, nlayers: int = 6,
                 dropout: float = 0.1, classes: int = 3):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # create the positional encoder
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        # define the encoder layers
        encoder_layers = TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model

        # we will not use the decoder
        # instead we will add a linear layer, another scaled dropout layer, and finally a classifier layer
        self.pre_classifier = torch.nn.Linear(d_model, d_model)
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(d_model, classes)
        self.logSoftmax = nn.LogSoftmax(dim=1)


    def forward(self, src: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
        Returns:
            output classifier label
        """
        # First we have to reshape the data
        # This needs to get fixed - or DSTL_dataset
        # The input is shaped like (batch, channel, len_slice) but
        # needs to be shaped like (batch, len_slice)

        # We should normalize the input weights by sqrt(d_model)
        src = src * math.sqrt(self.d_model)
        # ToDo: The positional encoder is changing the dimensions in a way I don't understand
        src = self.pos_encoder(src).squeeze()
        t_out = self.transformer_encoder(src)
        # ToDo: Perhaps with a working PE we need to use these to reshape
        #hidden_state = t_out[0]
        #pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(t_out)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        output = self.logSoftmax(output)
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        # ToDo: try the following change
        # pe = torch.zeros(max_len, 1, d_model)
        # pe[:, 0, 0::2] = torch.sin(position * div_term)
        # pe[:, 0, 1::2] = torch.cos(position * div_term)
        # try the following instead
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        # x = x + self.pe[:x.size(0)]
        x = x + self.pe[:, :x.size(0)]
        return self.dropout(x)
