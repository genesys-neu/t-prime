import math
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TransformerModel(nn.Module):

    def __init__(self, d_model: int = 512, seq_len: int = 64, nhead: int = 8, nlayers: int = 2,
                 dropout: float = 0.1, classes: int = 4, use_pos: bool = False):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.norm = nn.LayerNorm(d_model)
        # create the positional encoder
        self.use_positional_enc = use_pos
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        # define the encoder layers
        encoder_layers = TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model

        # we will not use the decoder
        # instead we will add a linear layer, another scaled dropout layer, and finally a classifier layer
        self.pre_classifier = torch.nn.Linear(d_model*seq_len, d_model)
        self.dropout = torch.nn.Dropout(0.5)
        self.classifier = torch.nn.Linear(d_model, classes)
        self.logSoftmax = nn.LogSoftmax(dim=1)


    def forward(self, src: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [batch_size, seq_len, features]
        Returns:
            output classifier label
        """
        #src = src * math.sqrt(self.d_model)
        src = self.norm(src)
        if self.use_positional_enc:
            src = self.pos_encoder(src).squeeze()
        t_out = self.transformer_encoder(src)
        t_out = torch.flatten(t_out, start_dim=1)
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
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # x = x + self.pe[:x.size(0)]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerModel_multiclass_transfer(nn.Module):

    def __init__(self, d_model: int = 512, seq_len: int = 64, nhead: int = 8, nlayers: int = 2, classes: int = 4):
        super(TransformerModel_multiclass_transfer, self).__init__()
        self.model_type = 'Transformer'

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.norm = nn.LayerNorm(d_model)
        # define the encoder layers
        encoder_layers = TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model

        # we will not use the decoder
        # instead we will add a linear layer, another scaled dropout layer, and finally a classifier layer
        self.pre_classifier = torch.nn.Linear(d_model*seq_len, d_model)
        # All previous layers will be frozen
        self.trainable_layer = torch.nn.Linear(d_model, d_model)
        self.dropout = torch.nn.Dropout(0.5)
        self.trainable_classifier = torch.nn.Linear(d_model, classes)
        self.sigmoid = nn.Sigmoid()


    def forward(self, src: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [batch_size, seq_len, features]
        Returns:
            output classifier label
        """
        #src = src * math.sqrt(self.d_model)
        src = self.norm(src)
        t_out = self.transformer_encoder(src)
        t_out = torch.flatten(t_out, start_dim=1)
        pooler = self.pre_classifier(t_out)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.trainable_layer(pooler)
        pooler = self.dropout(pooler)
        output = self.trainable_classifier(pooler)
        output = self.sigmoid(output)
        return output

class TransformerModel_multiclass(nn.Module):
    def __init__(self, d_model: int = 512, seq_len: int = 64, nhead: int = 8, nlayers: int = 2,
                 classes: int = 4):
        super(TransformerModel_multiclass, self).__init__()
        self.model_type = 'Transformer'

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.norm = nn.LayerNorm(d_model)
        # define the encoder layers
        encoder_layers = TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model

        # we will not use the decoder
        # instead we will add a linear layer, another scaled dropout layer, and finally a classifier layer
        self.pre_classifier = torch.nn.Linear(d_model*seq_len, d_model)
        self.dropout = torch.nn.Dropout(0.5)
        self.classifier = torch.nn.Linear(d_model, classes)
        self.sigmoid = nn.Sigmoid()


    def forward(self, src: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [batch_size, seq_len, features]
        Returns:
            output classifier label
        """
        #src = src * math.sqrt(self.d_model)
        src = self.norm(src)
        t_out = self.transformer_encoder(src)
        t_out = torch.flatten(t_out, start_dim=1)
        pooler = self.pre_classifier(t_out)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        output = self.sigmoid(output)
        return output

class TransformerModel_v2(nn.Module):

    def __init__(self, d_model: int = 512, seq_len: int = 64, nhead: int = 8, nlayers: int = 2,
                 dropout: float = 0.1, classes: int = 4, use_pos: bool = False):
        super(TransformerModel_v2, self).__init__()
        self.model_type = 'Transformer'

        self.norm = nn.LayerNorm(d_model)
        # create the positional encoder
        self.use_positional_enc = use_pos
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        # define [CLS] token to be used for classification
        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, d_model))
        # define the encoder layers
        encoder_layers = TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model

        # we will not use the decoder
        # instead we will add a linear layer, another scaled dropout layer, and finally a classifier layer
        self.pre_classifier = torch.nn.Linear(d_model, d_model)
        self.dropout = torch.nn.Dropout(0.2)
        self.classifier = torch.nn.Linear(d_model, classes)
        self.logSoftmax = nn.LogSoftmax(dim=1)


    def forward(self, src: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [batch_size, seq_len, features]
        Returns:
            output classifier label
        """

        # We normalize the input weights 
        cls_tokens = self.cls_token.repeat(src.size(0),1,1)
        src = torch.column_stack((cls_tokens, src))
        src = self.norm(src)
        if self.use_positional_enc:
            src = self.pos_encoder(src).squeeze()
        t_out = self.transformer_encoder(src)
        # get hidden state of the [CLS] token
        t_out = t_out[:,0,:].squeeze()
        pooler = self.pre_classifier(t_out)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        output = self.logSoftmax(output)
        return output
