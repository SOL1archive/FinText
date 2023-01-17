import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


from Data.Dataset import FinTextDataset
from Model.MainModel import FinTextModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = torch.load('./model-dir/')
model.eval()

with torch.no_grad():
    pass
