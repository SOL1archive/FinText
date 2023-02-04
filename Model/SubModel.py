import torch
import torch.nn as nn

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(GRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size, bias=True)
        
    def forward(self, x):
        x, _ = self.gru(x)
        x = self.fc(x)
        return x

class StdCNN2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, use_batch_norm=True):
        super(self, StdCNN2d).__init__()

        self.use_batch_norm = use_batch_norm

        self.conv2d = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size,
            stride=stride, 
            padding=padding
            )
        self.batch_norm = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=kernel_size)

    def forward(self, x):
        x = self.conv2d(x)
        if self.use_batch_norm:
            x = self.batch_norm(x)
        x = self.relu(x)
        x = self.max_pool(x)

class StdCNN3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, use_batch_norm=True):
        super(StdCNN3d, self).__init__()

        self.use_batch_norm = use_batch_norm

        self.conv3d = nn.Conv3d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size,
            stride=stride, 
            padding=padding
            )
        self.batch_norm = nn.BatchNorm3d(num_features=out_channels)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool3d(kernel_size=kernel_size)

    def forward(self, x):
        x = self.conv3d(x)
        if self.use_batch_norm:
            x = self.batch_norm(x)
        x = self.relu(x)
        x = self.max_pool(x)

        return x

def CNN_Layer(in_channels, out_channels, kernel_size, stride, padding):
    net = nn.Sequential(
        nn.Conv3d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size,
            stride=stride, 
            padding=padding
            ),
        nn.BatchNorm3d(num_features=out_channels),
        nn.ReLU(),
        nn.MaxPool3d(kernel_size=kernel_size)
    )

    return net