import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

from Dataloader.Dataset import FinTextDataset
from Dataloader.Dataloader import FinTextDataLoader
from Model.MainModel import FinTextModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset = Dataset()
dataloader = FinTextDataLoader(dataset)

model = FinTextModel()

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters())

for epoch in range(dataloader):
    for i, (input, labels) in enumerate(dataloader):
        optimizer.zero_grad()
