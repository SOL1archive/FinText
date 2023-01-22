import sys
import os

import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname('./..'))))

from Data.Dataset import FinTextDataset
from Data.DataLoader import FinTextDataLoader
from Model.MainModel import FinTextModel
import torch
import pandas as pd

def divide_chunks(lt, n):
     
    for i in range(0, len(lt), n):
        yield lt[i:i + n]

def to(tensor, device):
    tensor.to(device)

df = pd.read_pickle('../data-dir/data-df.pkl')
dataset = FinTextDataset(df)


dataset.save()