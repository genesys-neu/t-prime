import numpy as np
import torch
from torch import nn


# Define model
class Baseline_CNN1D(nn.Module):
    def __init__(self, numChannels=1, slice_len=4, classes=3, normalize=False):
        super(Baseline_CNN1D, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.numChannels = numChannels
        self.normalize = normalize
        self.norm = nn.LayerNorm(slice_len)
        # initialize first set of CONV => RELU => POOL layers
        self.conv1 = nn.Conv1d(in_channels=numChannels, out_channels=64, kernel_size=7)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        ##  initialize second set of CONV => RELU => POOL layers
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128,
                            kernel_size=3)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        ## initialize first (and only) set of FC => RELU layers

        # pass a random input
        rand_x = torch.Tensor(np.random.random((1, numChannels, slice_len)))
        output_size = torch.flatten(self.maxpool2(self.conv2(self.maxpool1(self.conv1(rand_x))))).shape
        self.fc1 = nn.Linear(in_features=output_size.numel(), out_features=256)
        self.relu3 = nn.ReLU()
        # initialize our softmax classifier
        self.fc2 = nn.Linear(in_features=256, out_features=classes)
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # x = x.reshape((x.shape[0], self.numChannels, x.shape[2]))   # CNN 1D expects a [N, Cin, L] size of data
        # pass the input through our first set of CONV => RELU =>
        # POOL layers
        if self.normalize:
            x = self.norm(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        ## pass the output from the previous layer through the second
        ## set of CONV => RELU => POOL layers
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        ## flatten the output from the previous layer and pass it
        ## through our only set of FC => RELU layers
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)
        # pass the output to our softmax classifier to get our output
        # predictions
        x = self.fc2(x)
        output = self.logSoftmax(x)
        # return the output predictions
        return output
