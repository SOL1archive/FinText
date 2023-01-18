import numpy as np
import pandas as pd

from sklearn.decomposition import TruncatedSVD, PCA, NMF

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from transformers import ElectraTokenizer, ElectraModel

from utils import *

class FinTextDataset(Dataset):
    def __init__(self, df, config=None):
        super(FinTextDataset, self).__init__()
        self.df = df
        #self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')

        if config == None:
            self.config = {
                "article_row_len": 5000,  # 조정될 필요 있음
                "community_row_len": 1000,
                "decomposition_method": "SVD",
                'bundle_size': 15
            }
        else:
            self.config = config

        def dim_fix(tensor, row_len):
            if tensor.shape[0] == row_len:
                return tensor
            elif tensor.shape[0] < row_len:
                increased_tensor = torch.zeros((row_len, tensor.shape[1]))
                increased_tensor[:tensor.shape[0], :] = tensor
                return increased_tensor
            else:
                method = self.config["decomposition_method"]
                if method == "SVD":
                    U, S, V = torch.svd(tensor)

                    reduced_S = S[:row_len]
                    reduced_U = U[:, :row_len]
                    reduced_V = V[:row_len, :]
                    reduced_tensor = torch.mm(
                        reduced_U, torch.mm(torch.diag(reduced_S), reduced_V.t())
                    )
                elif method == "PCA":
                    vectors_np = tensor.numpy()

                    pca = PCA(n_components=row_len)
                    reduced_tensor = pca.fit_transform(vectors_np)
                    reduced_tensor = torch.from_numpy(reduced_tensor)
                elif method == "NMF":
                    vectors_np = tensor.numpy()

                    nmf = NMF(n_components=row_len)
                    reduced_tensor = nmf.fit_transform(vectors_np)
                    reduced_tensor = torch.from_numpy(reduced_tensor)
                else:
                    raise RuntimeError

                return reduced_tensor
            
        def embed_text(text_lt, tokenizer, model):
            article_tensor_lt = []
            for text in text_lt:
                base_vector = torch.tensor(tokenizer.encode(text)).unsqueeze(0).to(self.device)
                embedded_matrix = torch.tensor(model(base_vector)[0][0]).to(self.device)
                article_tensor_lt.append(embedded_matrix)

            return torch.cat(article_tensor_lt, dim=0)
        
        self.feature_df = self.df.drop('Label', axis=1)

        ko_tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
        ko_model = ElectraModel.from_pretrained("monologg/koelectra-base-v3-discriminator").to(self.device)
        kc_tokenizer = ElectraTokenizer.from_pretrained("beomi/KcELECTRA-base-v2022")
        kc_model = ElectraModel.from_pretrained("beomi/KcELECTRA-base-v2022").to(self.device)

        row_dict = dict()
        row_dict['article_matrix'] = []
        row_dict['community_matrix'] = []
        row_dict['community_metric_index'] = []
        row_dict['price_vector'] = []

        for _, period in self.feature_df.iterrows():
            article_matrix = embed_text(
                period["ArticleText"], ko_tokenizer, ko_model
            )
            article_matrix = dim_fix(article_matrix, self.config['article_row_len'])

            community_matrix = embed_text(
                period["CommunityText"], kc_tokenizer, kc_model
            )
            community_matrix = dim_fix(community_matrix, self.config['community_row_len'])
            
            community_metric_index = torch.tensor(period["MetricIndex"])

            price_vector = torch.tensor(
                [period["Open"], period["High"], period["Low"], period["Close"]]
            )

            row_dict['article_matrix'].append(article_matrix)
            row_dict['community_matrix'].append(community_matrix)
            row_dict['community_metric_index'].append(community_metric_index)
            row_dict['price_vector'].append(price_vector)
        
        def make_chunk_and_stack(data_lt):
            row_lt = []
            for row in divide_chunks(data_lt, self.config['bundle_size']):
                row_lt.append(
                    torch.stack(row)
                    )

            return row_lt

        feature_dict = dict()
        for name, total_row in row_dict.items():
            feature_dict[name] = make_chunk_and_stack(total_row)
        self.feature_df = pd.DataFrame(feature_dict)
        self.target_tensor = torch.tensor(
            pd.get_dummies(self.df['Label']).values[0:-1:self.config['bundle_size']]
        )

    def __len__(self):
        return len(self.feature_df)

    def __getitem__(self, index):
        return (
            self.feature_df.loc(index), 
            self.target_tensor[index]
            )
