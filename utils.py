import torch
import pandas as pd

from Data.Dataset import FinTextDataset    

def divide_chunks(lt, n):
     
    for i in range(0, len(lt), n):
        yield lt[i:i + n]

def to(tensor, device):
    tensor.to(device)
    
def concat_dataset(df_list):
    dataset_lt = []
    for df in df_list:
        dataset_lt.append(FinTextDataset(df))
    
    dataset = dataset_lt[0]
    for dataset_item in dataset_lt[1:]:
        dataset.feature_df = pd.concat(
            [dataset.feature_df, dataset_item.feature_df]
        )
        dataset.target_tensor = torch.concat(
            [dataset.target_tensor, dataset_item.target_tensor]
        )
    
    return dataset
