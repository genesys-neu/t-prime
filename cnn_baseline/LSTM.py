import math
import torch
from torch import nn

class ComplexDataClassificationModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(ComplexDataClassificationModel, self).__init__()
        
        # LSTM layers
        self.lstm_layers = nn.LSTM(input_size=input_size, 
                                   hidden_size=hidden_size, 
                                   num_layers=num_layers, 
                                   batch_first=True)
        
        # Dense layer for final classification
        self.dense = nn.Linear(hidden_size, num_classes)
        
    def forward(self, amplitude, phase):
        # Concatenate amplitude and phase vectors along the last dimension
        complex_input = torch.cat([amplitude, phase], dim=-1)
        
        # LSTM input shape should be (batch_size, sequence_length, input_size)
        # The time steps are assumed to be the second dimension in the input tensors
        lstm_output, _ = self.lstm_layers(complex_input)
        
        # Extract the output of the second LSTM layer after all time steps
        final_output = lstm_output[:, -1, :]
        
        # Pass the final output through the dense layer for classification
        output = self.dense(final_output)
        
        return output