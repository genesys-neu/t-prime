import math
import torch
from torch import nn


class LSTM_ap(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.8):
        super(LSTM_ap, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Reshape the output for the fully connected layer
        out = out[:, -1, :]  # Take only the last output in the sequence
        out = self.fc(out)

        return out