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
                return F.pad(tensor, (row_len, tensor.shape[1]), value=0)
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
                base_vector = torch.tensor(tokenizer.encode(text)).unsqueeze(0)
                embedded_matrix = torch.tensor(model(base_vector)[0][0])
                article_tensor_lt.append(embedded_matrix)

            return torch.cat(article_tensor_lt, dim=0)
        
        self.feature_df = self.df.drop('Label', axis=1)

        ko_tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
        ko_model = ElectraModel.from_pretrained("monologg/koelectra-base-v3-discriminator")
        kc_tokenizer = ElectraTokenizer.from_pretrained("beomi/KcELECTRA-base-v2022")
        kc_model = ElectraModel.from_pretrained("beomi/KcELECTRA-base-v2022")

        feature_lt = []
        for period in self.feature_df.iterrows():
            article_matrix = embed_text(
                period["ArticleText"], ko_tokenizer, ko_model
            )
            article_matrix = dim_fix(article_matrix)

            community_matrix = embed_text(
                period["CommunityText"], kc_tokenizer, kc_model
            )
            community_matrix = dim_fix(community_matrix)
            community_metric_index = period["MetricIndex"]

            price_vector = torch.tensor(
                [period["Open"], period["High"], period["Low"], period["Close"]]
            )

            row_tensor = torch.cat(
                [article_matrix, community_matrix, community_metric_index, price_vector],
                dim=1
            )

            feature_lt.append(row_tensor)
        
        def make_chunk_and_stack(data_lt):
            row_lt = []
            for row in divide_chunks(feature_lt, self.config['bundle_size']):
                row_lt.append(
                    torch.stack(row)
                    )

            return torch.stack(row_lt)

        self.feature_tensor = make_chunk_and_stack(feature_lt)
        self.target_tensor = make_chunk_and_stack(
            pd.get_dummies(self.df['Label']).values
        )

    def __len__(self):
        return len(self.feature_tensor)

    def __getitem__(self, index):
        return (self.feature_tensor[index], self.target_tensor[index])
