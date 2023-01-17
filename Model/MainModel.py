import torch
import torch.nn as nn
import torch.nn.functional as F

from Model.utils import CNN_Layer

class FinTextModel(nn.Module):
    def __init__(self):
        super(FinTextModel, self).__init__()

        # Neural Networks
        self.community_cnn = CNN_Layer(
            in_channels=1, out_channels=10, kernel_size=(3, 3, 9), stride=1, padding=(4, 1)
        )

        self.article_cnn = CNN_Layer(
            in_channels=1, out_channels=10, kernel_size=(3, 3, 9), stride=1, padding=(4, 1)
        )

        self.community_metric_ffn = nn.Sequential(
            nn.Linear(in_features=1, out_features=1, bias=False),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
        )

        self.community_ffn = nn.Sequential(
            nn.Linear(in_features=1, out_features=1), nn.ReLU()
        )

        self.gru = nn.Sequential(
            nn.GRU(input_size=4, hidden_size=10, num_layers=2)
        )

        self.hidden_init_state = torch.zeros()

        self.total_ffn = nn.Sequential(
            nn.Linear(in_features=10000, out_features=10000),
            nn.ReLU(),
            nn.Linear(in_features=10000, out_features=7000),
            nn.ReLU(),
            nn.Linear(in_features=7000, out_features=5000),
            nn.ReLU(),
        )

        self.softmax = nn.Sequential(
            nn.Linear(in_features=5000, out_features=4), nn.Softmax(dim=10)
        )

    def forward(self, x):
        # Slicing Tensor
        article_tensor = x[0]
        community_tensor = x[1]
        community_metric_index = x[2]
        price_index = x[3]

        # In Neural Network
        article_tensor = self.article_cnn(article_tensor)

        community_tensor = self.community_cnn(community_tensor)
        community_metric_index = self.community_metric_ffn(community_metric_index)
        
        price_index = self.gru(price_index)

        total_out = torch.cat(
            [
                nn.Flatten(article_tensor), 
                nn.Flatten(community_tensor),
                community_metric_index,
                price_index
            ], dim=0
        )

        total_out = self.total_ffn(total_out)
        total_out = self.softmax(total_out)

        return total_out
        