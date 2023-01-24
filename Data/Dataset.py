import os
from pprint import pprint

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, NMF

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from transformers import ElectraTokenizer, ElectraModel

LABEL_NUM = 2

def to(tensor_dict, device):
    for feature in tensor_dict.keys():
        tensor_dict[feature] = tensor_dict[feature].to(device)
    
    return tensor_lt

def dict_index(input_dict, idx):
    result = dict()
    for key in input_dict.keys():
        result[key] = input_dict[key][idx]

    return result

class FinTextDataset(Dataset):
    def __init__(self, df, **config):
        super(FinTextDataset, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #device = torch.device("cpu")

        default_config = {
            "community_row_len": 5000,
            "decomposition_method": "Dull",
            "bundle_size": 14,
        }

        for key in default_config.keys():
            if key not in config.keys():
                config[key] = default_config[key]

        self.config = config

        if 'feature_dict' in self.config.keys() and 'target_tensor' in self.config.keys():
            self.feature_dict = self.config['feature_dict']
            self.target_tensor = self.config['target_tensor']
            self.config.pop('feature_dict')
            self.config.pop('target_tensor')
            return

        if df is None:
            raise RuntimeError
        elif type(df) == str:
            os.chdir(f'{df}')
            feature_lt = []
            for file in os.listdir():
                if file.startswith('feature_') and file.endswith('.pt'):
                    feature_lt.append(
                        file[file.find('_') + 1:file.rfind('.pt')]
                    )
            self.feature_dict = dict()
            for feature in feature_lt:
                self.feature_dict[feature] = torch.load(f'./feature_{feature}.pt')

            self.target_tensor = torch.load('./target.pt')
            os.chdir('../../..')
            return

        def dim_fix(tensor, row_len):
            if tensor.shape[0] == row_len:
                return tensor
            elif tensor.shape[0] < row_len:
                increased_tensor = torch.zeros(row_len, tensor.shape[1]).to(device)
                increased_tensor[:tensor.shape[0], :] = tensor

                return increased_tensor
            else:
                method = self.config["decomposition_method"]
                tensor = tensor.t()
                if method == "SVD":
                    U, S, V = torch.pca_lowrank(tensor, q=row_len)
                    print(V.shape)
                    #reduced_S = S[:row_len]
                    #reduced_U = U[:, :row_len]
                    #reduced_V = V[:row_len, :]
                    reduced_tensor = torch.mm(tensor, V[:, :row_len])
                elif method == "PCA":
                    vectors_np = tensor.cpu().numpy()

                    pca = PCA(n_components=row_len)
                    reduced_tensor = pca.fit_transform(vectors_np)
                    reduced_tensor = torch.from_numpy(reduced_tensor)
                elif method == "NMF":
                    vectors_np = tensor.cpu().numpy()

                    nmf = NMF(n_components=row_len)
                    reduced_tensor = nmf.fit_transform(vectors_np)
                    reduced_tensor = torch.from_numpy(reduced_tensor)
                elif method == 'Dull':
                    reduced_tensor = tensor[:, :row_len]
                else:
                    raise RuntimeError

                reduced_tensor = reduced_tensor.t()

                return reduced_tensor

        def embed_text(text_lt, tokenizer, model):
            article_tensor_lt = []
            for text in text_lt:
                if type(text) == str:
                    base_vector = (
                        tokenizer.encode(
                            text, 
                            return_tensors='pt',
                            max_length=512,
                            truncation=True
                        ).to(device)
                    )
                    try:
                        embedded_matrix = model(base_vector)
                    except RuntimeError:
                        continue
                    embedded_matrix = torch.tensor(embedded_matrix[0][0]).to(device)
                    article_tensor_lt.append(embedded_matrix)
                    
                    del base_vector

            total_tensor = torch.cat(article_tensor_lt, dim=0)
            for tensor in article_tensor_lt:
                del tensor
            torch.cuda.empty_cache()

            return total_tensor

        feature_df = df.drop("Label", axis=1)

        kc_tokenizer = ElectraTokenizer.from_pretrained(
            "beomi/KcELECTRA-base-v2022"
        )
        kc_model = ElectraModel.from_pretrained(
            "beomi/KcELECTRA-base-v2022"
        ).to(device)

        row_dict = dict()
        row_dict["community_tensor"] = []
        row_dict["community_metric_index"] = []
        row_dict["price_index"] = []

        for _, period in feature_df.iterrows():

            community_tensor = embed_text(
                period["CommunityText"], kc_tokenizer, kc_model
            )
            non_singular_community = dim_fix(
                community_tensor, self.config["community_row_len"]
            )
            community_tensor = non_singular_community

            non_singular_metric = torch.tensor(period["MetricIndex"])
            community_metric_index = torch.zeros(2200)
            community_metric_index[:non_singular_metric.shape[0]] = non_singular_metric

            price_index = torch.tensor(
                [period["Open"], period["High"], period["Low"], period["Close"]]
            )

            row_dict["community_tensor"].append(community_tensor)
            row_dict["community_metric_index"].append(community_metric_index)
            row_dict["price_index"].append(price_index)
        
        del kc_tokenizer
        del kc_model
        torch.cuda.empty_cache()

        def make_chunk_and_stack(data_lt):
            def divide_chunks(lt, n):     
                for i in range(0, len(lt), n):
                    yield lt[i:i + n]
            
            row_lt = []
            for row in divide_chunks(data_lt, self.config["bundle_size"]):
                row_lt.append(torch.stack(row))

            return row_lt

        feature_dict = dict()
        for name, total_row in row_dict.items():
            print(name)
            for i, row in enumerate(total_row):
                total_row[i] = row.to('cpu')
            feature_dict[name] = make_chunk_and_stack(total_row)
            
        for feature in feature_dict:
            feature_dict[feature] = torch.stack(
                feature_dict[feature]
            )

        self.feature_dict = feature_dict
        
        self.target_tensor = F.one_hot(
            torch.tensor(df["Label"].values), 
            num_classes=LABEL_NUM
        )[0 : -1 : self.config["bundle_size"]]

    def to(self, device):
        self.target_tensor.to(device)
        for feature in self.feature_dict:
            self.feature_dict[feature].to(device)

    def train_test_split(self, train_size=0.80):
        train_index, test_index = train_test_split(
            range(len(self.target_tensor)), train_size=train_size
        )

        train_feature_dict = dict()
        for feature in self.feature_dict.keys():
            train_feature_dict[feature] = self.feature_dict[feature][train_index]
        train_target = self.target_tensor[train_index]
        
        test_feature_dict = dict()
        for feature in self.feature_dict.keys():
            test_feature_dict[feature] = self.feature_dict[feature][test_index]
        test_target = self.target_tensor[test_index]

        train_dataset = FinTextDataset(
            df=None, 
            feature_dict=train_feature_dict,
            target_tensor=train_target,
            **self.config
        )
        test_dataset = FinTextDataset(
            df=None,
            feature_dict=test_feature_dict,
            target_tensor=test_target,
            **self.config
        )

        return train_dataset, test_dataset

    def save(self, path='dataset'):
        os.mkdir(f'{path}')
        os.chdir(f'{path}')
        for feature in self.feature_dict.keys():
            torch.save(self.feature_dict[feature], f'feature_{feature}.pt')

        torch.save(self.target_tensor, 'target.pt')
        os.chdir('..')

    def __len__(self):
        return len(self.target_tensor)

    def __getitem__(self, index):
        return (
            dict_index(self.feature_dict, index), 
            self.target_tensor[index]
        )

def concat_dataset(dataset_lt):
    dataset = dataset_lt[0]

    for feature in dataset.feature_dict.keys():
        total_feature_lt = []
        for dataset_item in dataset_lt[:4]:
            total_feature_lt.append(
                dataset_item.feature_dict[feature]
            )
        dataset.feature_dict[feature] = torch.concat(total_feature_lt)

    dataset.target_tensor = torch.concat(
        [
            dataset.target_tensor, 
        ] +
        [
            dataset_item.target_tensor for dataset_item in dataset_lt[1:4]
        ]
    )

    return dataset

def concat_df_dataset(df_list):
    dataset_lt = []
    for df in df_list:
        dataset_lt.append(FinTextDataset(df))
    
    dataset = concat_dataset(dataset_lt)
    
    return dataset
