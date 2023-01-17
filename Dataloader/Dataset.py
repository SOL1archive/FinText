import numpy as np
import pandas as pd

from sklearn.decomposition import TruncatedSVD, PCA, NMF

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from transformers import ElectraTokenizer, ElectraModel

class FinTextDataset(Dataset):
    def __init__(self, df, config=None):
        super(FinTextDataset, self).__init__()
        self.df = df

        if config == None:
            self.config = {
                "article_row_len": 5000,  # 조정될 필요 있음
                "community_row_len": 1000,
                "decomposition_method": "SVD",
            }
        else:
            self.config = config

        def dim_fix(self, tensor, row_len):
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

            
        def embed_text(self, text_lt, tokenizer, model):
            article_tensor_lt = []
            for text in text_lt:
                base_vector = torch.tensor(tokenizer.encode(text)).unsqueeze(0)
                embedded_matrix = torch.tensor(model(base_vector)[0][0])
                article_tensor_lt.append(embedded_matrix)

            return torch.cat(article_tensor_lt, dim=0)
        
        self.ko_tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
        self.ko_model = ElectraModel.from_pretrained("monologg/koelectra-base-v3-discriminator")
        self.kc_tokenizer = ElectraTokenizer.from_pretrained("beomi/KcELECTRA-base-v2022")
        self.kc_model = ElectraModel.from_pretrained("beomi/KcELECTRA-base-v2022")

        x_dict = dict()
        x_dict["article_matrix"] = []
        x_dict["community_matrix"] = []
        x_dict["community_metric_index"] = []
        x_dict["price_vector"] = []
        for period in self.df.iterrows():
            article_matrix = embed_text(
                period["ArticleText"], self.ko_tokenizer, self.ko_model
            )
            community_matrix = embed_text(
                period["CommunityText"], self.kc_tokenizer, self.ko_model
            )
            community_metric_index = period["MetricIndex"]
            price_vector = torch.tensor(
                [period["Open"], period["High"], period["Low"], period["Close"]]
            )

            x_dict["article_matrix"].append(article_matrix)
            x_dict["community_matrix"].append(community_matrix)
            x_dict["community_metric_index"].append(community_metric_index)
            x_dict["price_vector"].append(price_vector)

        article_tensor = torch.cat(x_dict["article_matrix"], dim=0)
        article_tensor = self.dim_fix(article_tensor, self.config["article_row_len"])
        community_tensor = torch.cat(x_dict["community_matrix"], dim=0)
        community_tensor = self.dim_fix(
            community_tensor, self.config["community_row_len"]
        )

        community_metric_index = torch.tensor(x_dict["community_metric_index"]).view(-1, 1)
        market_index = torch.tensor(x_dict['price_vector'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        x = self.df.iloc[index, :-1]
        y = self.df.iloc[index, -1]
        return x, y