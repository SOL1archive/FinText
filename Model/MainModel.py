import torch
import torch.nn as nn

from Model.SubModel import *

class FinTextModel(nn.Module):
    def __init__(self):
        super(FinTextModel, self).__init__()

        # Neural Networks
        self.community_cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=1, 
                out_channels=10, 
                kernel_size=(3, 9, 768),
                stride=1, 
                padding=(1, 4, 0)
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 9)),
            
            nn.Conv2d(
                in_channels=10, 
                out_channels=20, 
                kernel_size=(9, 3),
                stride=1, 
                padding=(4, 1, 0)
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(9, 3)),
        )

        self.community_metric_ffn = nn.Sequential(
            nn.Linear(in_features=2200, out_features=14, bias=False),
            #nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
        )

        self.community_ffn = nn.Sequential(
            nn.Linear(in_features=1, out_features=1), 
            nn.ReLU()
        )

        self.gru = GRU(
            input_size=4, 
            hidden_size=10, 
            output_size=4, 
            num_layers=5
        )

        self.flatten = nn.Flatten()
            
        self.total_ffn = nn.Sequential(
            nn.Linear(in_features=103952, out_features=10000), # 다른 층의 출력에 맞게 조정되어야 함.
            nn.ReLU(),
            nn.Linear(in_features=10000, out_features=7000),
            nn.ReLU(),
            nn.Linear(in_features=7000, out_features=5000),
            nn.ReLU(),
        )

        self.softmax = nn.Sequential(
            nn.Linear(in_features=5000, out_features=10), 
            nn.Softmax(dim=2)
        )

    def forward(self, x):
        # Slicing Tensor
        community_tensor = x['community_tensor']
        community_metric_index = x['community_metric_index']
        price_index = x['price_index']

        # In Neural Network
        community_tensor = self.community_cnn(community_tensor)
        community_tensor = self.flatten(community_tensor)
        community_tensor = community_tensor.view(-1, 1)

        community_metric_index = self.community_metric_ffn(community_metric_index)
        community_metric_index = self.flatten(community_metric_index)
        community_metric_index = community_metric_index.view(-1, 1)

        price_index = self.gru(price_index)
        price_index = self.flatten(price_index)
        price_index = price_index.view(-1, 1)

        r'''
        output tensor dimension 계산공식:
        python: 파이썬 인터프리터에 대입해서 쉽게 연산하고 싶을 떄 사용)
            int((input_size - kernel_size + 2 * padding) / stride) + 1
        LaTeX: 시각적으로 수식을 볼때 사용)
            $\frac{input_size - kernel_size + 2padding}{stride} + 1$
        '''

        total_out = torch.cat(
            [
                community_tensor,
                community_metric_index,
                price_index
            ], dim=0
        ).view(1, -1)

        total_out = self.total_ffn(total_out)
        
        total_out = self.softmax(total_out)

        return total_out
        