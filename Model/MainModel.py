import numpy as np
import pandas as pd

from sklearn.decomposition import TruncatedSVD, PCA, NMF

import torch
import torch.nn as nn
import torch.functional as F

from transformers import ElectraTokenizer, ElectraModel

from utils import CNN_Layer

class FinTextModel(nn.Module):
    def __init__(self, hyper_parameters=None):
        super(FinTextModel, self).__init__()

        if hyper_parameters == None:
            self.hyper_parameters = {
                'community': {
                    'cnn_out_channels': 10,
                    'kernel_size': (9, 3),
                    'stride': 1,
                    'same': True
                },
                'article': {
                    'cnn_out_channels': 10,
                    'kernel_size': (9, 3),
                    'stride': 1,
                    'same': True
                },
                'max_row_len': 10000,
                'decomposition_method': 'SVD'
            }
        else:
            self.hyper_parameters = hyper_parameters

        self.ko_tokenizer = ElectraTokenizer.from_pretrained('monologg/koelectra-base-v3-discriminator')
        self.ko_model = ElectraModel.from_pretrained('monologg/koelectra-base-v3-discriminator')
        self.kc_tokenizer = ElectraTokenizer.from_pretrained('beomi/KcELECTRA-base-v2022')
        self.kc_model = ElectraModel.from_pretrained('beomi/KcELECTRA-base-v2022')

        self.community_cnn = CNN_Layer(
            in_channels=1,
            out_channels=self.hyper_parameters['community']['cnn_out_channels'],
            kernel_size=self.hyper_parameters['community']['kernel_size'],
            stride=self.hyper_parameters['community']['stride'],
            padding=(
                self.hyper_parameters['community']['kernel_size'][0] // 2,
                self.hyper_parameters['community']['kernel_size'][1] // 2,
                ) if self.hyper_parameters['community']['same'] else 0
        )

        self.article_cnn = CNN_Layer(
            in_channels=1,
            out_channels=self.hyper_parameters['article']['cnn_out_channels'],
            kernel_size=self.hyper_parameters['article']['kernel_size'],
            stride=self.hyper_parameters['article']['stride'],
            padding=(
                self.hyper_parameters['article']['kernel_size'][0] // 2,
                self.hyper_parameters['article']['kernel_size'][1] // 2,
                ) if self.hyper_parameters['article']['same'] else 0
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

        self.softmax = nn.Softmax(dim=10)

    def embed_text(text_lt, tokenizer, model):
        article_tensor_lt = []
        for text in text_lt:
            base_vector = torch.tensor(tokenizer.encode(text)).unsqueeze(0)
            embedded_matrix = torch.tensor(model(base_vector)[0][0])
            article_tensor_lt.append(
                embedded_matrix
            )
        
        return torch.cat(article_tensor_lt, dim=0)

    def dim_reduction(self, tensor):
        max_row = self.hyper_parameters['max_row_len']
        if tensor.shape[0] <= max_row:
            return tensor
        else:
            method = self.hyper_parameters['decomposition_method']
            if method == 'SVD':
                U, S, V = torch.svd(tensor)

                reduced_S = S[:max_row]
                reduced_U = U[:, :max_row]
                reduced_V = V[:max_row, :]
                reduced_tensor = torch.mm(reduced_U, torch.mm(torch.diag(reduced_S), reduced_V.t()))
            elif method == 'PCA':
                vectors_np = tensor.numpy()

                pca = PCA(n_components=10)
                reduced_tensor = pca.fit_transform(vectors_np)
                reduced_tensor = torch.from_numpy(reduced_tensor)
            elif method == 'NMF':
                vectors_np = tensor.numpy()

                nmf = NMF(n_components=max_row)
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
        article_tensor = self.dim_reduction(article_tensor)
        community_tensor = torch.cat(x_dict['community_matrix'], dim=0)
        community_tensor = self.dim_reduction(community_tensor)
    
        community_metric_index = torch.tensor(x_dict['community_metric_index']).view(-1, 1)

        # In Neural Network
        community_metric_index = self.community_metric_ffn(community_metric_index)
