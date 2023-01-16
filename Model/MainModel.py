import numpy as np
import pandas as pd

from sklearn.decomposition import TruncatedSVD, PCA, NMF

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import ElectraTokenizer, ElectraModel

from utils import CNN_Layer

class FinTextModel(nn.Module):
    def __init__(self, config=None):
        super(FinTextModel, self).__init__()

        if config == None:
            self.config = {
                'article_row_len': 100000, # 조정될 필요 있음
                'community_row_len': 100000,
                'decomposition_method': 'SVD'
            }
        else:
            self.config = config

        # Pre-trained Model
        self.ko_tokenizer = ElectraTokenizer.from_pretrained('monologg/koelectra-base-v3-discriminator')
        self.ko_model = ElectraModel.from_pretrained('monologg/koelectra-base-v3-discriminator')
        self.kc_tokenizer = ElectraTokenizer.from_pretrained('beomi/KcELECTRA-base-v2022')
        self.kc_model = ElectraModel.from_pretrained('beomi/KcELECTRA-base-v2022')

        # Neural Networks
        self.community_cnn = CNN_Layer(
            in_channels=1,
            out_channels=10,
            kernel_size=(9, 3),
            stride=1,
            padding=(4, 1)
        )

        self.article_cnn = CNN_Layer(
            in_channels=1,
            out_channels=10,
            kernel_size=(9, 3),
            stride=1,
            padding=(4, 1)
        )

        self.community_metric_ffn = nn.Sequential(
            nn.Linear(in_features=1, out_features=1, bias=False),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU()
        )

        self.community_ffn = nn.Sequential(
            nn.Linear(in_features=1, out_features=1),
            nn.ReLU()
        )

        self.total_ffn = nn.Sequential(
            nn.Linear(in_features=10000, out_features=10000),
            nn.ReLU(),
            nn.Linear(in_features=10000, out_features=7000),
            nn.ReLU(),
            nn.Linear(in_features=7000, out_features=5000),
            nn.ReLU(),
        )

        self.softmax = nn.Sequential(
            nn.Linear(in_features=5000, out_features=4),
            nn.Softmax(dim=10)
        )

    def embed_text(self, text_lt, tokenizer, model):
        article_tensor_lt = []
        for text in text_lt:
            base_vector = torch.tensor(tokenizer.encode(text)).unsqueeze(0)
            embedded_matrix = torch.tensor(model(base_vector)[0][0])
            article_tensor_lt.append(
                embedded_matrix
            )
        
        return torch.cat(article_tensor_lt, dim=0)

    def dim_fix(self, tensor, row_len):
        if tensor.shape[0] == row_len:
            return tensor
        elif tensor.shape[0] < row_len:
            return F.pad(tensor, (row_len, tensor.shape[1]), value=0)
        else:
            method = self.config['decomposition_method']
            if method == 'SVD':
                U, S, V = torch.svd(tensor)

                reduced_S = S[:row_len]
                reduced_U = U[:, :row_len]
                reduced_V = V[:row_len, :]
                reduced_tensor = torch.mm(reduced_U, torch.mm(torch.diag(reduced_S), reduced_V.t()))
            elif method == 'PCA':
                vectors_np = tensor.numpy()

                pca = PCA(n_components=row_len)
                reduced_tensor = pca.fit_transform(vectors_np)
                reduced_tensor = torch.from_numpy(reduced_tensor)
            elif method == 'NMF':
                vectors_np = tensor.numpy()

                nmf = NMF(n_components=row_len)
                reduced_tensor = nmf.fit_transform(vectors_np)
                reduced_tensor = torch.from_numpy(reduced_tensor)
            else:
                raise RuntimeError

            return reduced_tensor

    def forward(self, x):
        x_dict = dict()
        x_dict['article_matrix'] = []
        x_dict['community_matrix'] = []
        x_dict['community_metric_index'] = []
        x_dict['price_vector'] = []
        for period in x:
            article_matrix = self.embed_text(period['ArticleText'], self.ko_tokenizer, self.ko_model)
            community_matrix = self.embed_text(period['CommunityText'], self.kc_tokenizer, self.ko_model)
            community_metric_index = period['MetricIndex']
            price_vector = torch.tensor([period['Open'], period['High'], period['Low'], period['Close']])
            
            x_dict['article_matrix'].append(article_matrix)
            x_dict['community_matrix'].append(community_matrix)
            x_dict['community_metric_index'].append(community_metric_index)
            x_dict['price_vector'].append(price_vector)

        article_tensor = torch.cat(x_dict['article_matrix'], dim=0)
        article_tensor = self.dim_fix(article_tensor, self.config['article_row_len'])
        community_tensor = torch.cat(x_dict['community_matrix'], dim=0)
        community_tensor = self.dim_fix(community_tensor, self.config['community_row_len'])
    
        community_metric_index = torch.tensor(x_dict['community_metric_index']).view(-1, 1)

        # In Neural Network
        article_tensor = self.article_cnn(article_tensor)

        community_tensor = self.community_cnn(community_tensor)
        community_metric_index = self.community_metric_ffn(community_metric_index)

        
