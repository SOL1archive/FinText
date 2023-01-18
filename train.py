import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from Data.Dataset import FinTextDataset
from Data.DataLoader import FinTextDataLoader
from Model.MainModel import FinTextModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'
writer = SummaryWriter()

dataset = Dataset()
dataloader = FinTextDataLoader(dataset)

model = FinTextModel().to(device)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = Adam(model.parameters())

for epoch in range(dataloader):
    for i, (input, labels) in enumerate(dataloader):
        # Forward
        input = input.to(device)
        output = model(input)
        loss = criterion(output, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

writer.flush()