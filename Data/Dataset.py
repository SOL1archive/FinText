import random

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD, PCA, NMF

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from transformers import ElectraTokenizer, ElectraModel

from utils import *


class FinTextDataset(Dataset):
    def __init__(self, df, **config):
        super(FinTextDataset, self).__init__()
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device("cpu")

        default_config = {
            "article_row_len": 5000,  # 조정될 필요 있음
            "community_row_len": 1000,
            "decomposition_method": "SVD",
            "bundle_size": 15,
        }

        for key in default_config.keys():
            if key not in config.keys():
                config[key] = default_config[key]

        self.config = config

        if 'feature_df' in self.config.keys() and 'target_tensor' in self.config.keys():
            self.feature_df = self.config['feature_df']
            self.target_tensor = self.config['target_tensor']
            self.config.pop('feature_df')
            self.config.pop('target_tensor')
            return

        if df == None:
            raise RuntimeError

        def dim_fix(tensor, row_len):
            if tensor.shape[0] == row_len:
                return tensor
            elif tensor.shape[0] < row_len:
                increased_tensor = torch.zeros((row_len, tensor.shape[1]))
                increased_tensor[: tensor.shape[0], :] = tensor
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
                base_vector = (
                    torch.tensor(tokenizer.encode(text)).unsqueeze(0).to(device)
                )
                embedded_matrix = torch.tensor(model(base_vector)[0][0]).to(device)
                article_tensor_lt.append(embedded_matrix)

            return torch.cat(article_tensor_lt, dim=0)

        feature_df = df.drop("Label", axis=1)

        ko_tokenizer = ElectraTokenizer.from_pretrained(
            "monologg/koelectra-base-v3-discriminator"
        )
        ko_model = ElectraModel.from_pretrained(
            "monologg/koelectra-base-v3-discriminator"
        ).to(device)
        kc_tokenizer = ElectraTokenizer.from_pretrained("beomi/KcELECTRA-base-v2022")
        kc_model = ElectraModel.from_pretrained("beomi/KcELECTRA-base-v2022").to(
            device
        )

        row_dict = dict()
        row_dict["article_tensor"] = []
        row_dict["community_matrix"] = []
        row_dict["community_metric_index"] = []
        row_dict["price_index"] = []

        for _, period in feature_df.iterrows():
            article_tensor = embed_text(period["ArticleText"], ko_tokenizer, ko_model)
            article_tensor = dim_fix(article_tensor, self.config["article_row_len"])

            community_matrix = embed_text(
                period["CommunityText"], kc_tokenizer, kc_model
            )
            community_matrix = dim_fix(
                community_matrix, self.config["community_row_len"]
            )

            community_metric_index = torch.tensor(period["MetricIndex"])

            price_index = torch.tensor(
                [period["Open"], period["High"], period["Low"], period["Close"]]
            )

            row_dict["article_tensor"].append(article_tensor)
            row_dict["community_matrix"].append(community_matrix)
            row_dict["community_metric_index"].append(community_metric_index)
            row_dict["price_index"].append(price_index)

        def make_chunk_and_stack(data_lt):
            row_lt = []
            for row in divide_chunks(data_lt, self.config["bundle_size"]):
                row_lt.append(torch.stack(row))

            return row_lt

        feature_dict = dict()
        for name, total_row in row_dict.items():
            feature_dict[name] = make_chunk_and_stack(total_row)
        self.feature_df = pd.DataFrame(feature_dict)
        self.target_tensor = torch.tensor(
            pd.get_dummies(self.df["Label"]).values[0 : -1 : self.config["bundle_size"]]
        )

    def to(self, device):
        self.target_tensor.to(device)
        for _, row in self.feature_df.iterrows():
            for item in row:
                item.to(device)

    def train_test_split(self, train_size=0.80):
        train_index, test_index = train_test_split(
            range(len(self.feature_df)), train_size=train_size
        )

        train_feature = self.feature_df.iloc[train_index]
        train_target = self.target_tensor[train_index]
        test_feature = self.feature_df.iloc[test_index]
        test_target = self.target_tensor[test_index]

        train_dataset = FinTextDataset(
            df=None, 
            feature_df=train_feature,
            target_tensor=train_target,
            **self.config
        )
        test_dataset = FinTextDataset(
            df=None,
            feature_df=test_feature,
            target_tensor=test_target,
            **self.config
        )

        return train_dataset, test_dataset

    def __len__(self):
        return len(self.feature_df)

    def __getitem__(self, index):
        return (self.feature_df.iloc[index], self.target_tensor[index])
