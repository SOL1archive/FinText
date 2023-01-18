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
    for i, (input_tensor, labels) in enumerate(dataloader):
        # Forward
        input_tensor = input_tensor.to(device)
        output_tensor = model(input_tensor)
        loss = criterion(output_tensor, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

writer.flush()

try:
    print('Press Ctrl + C to turn off the TensorBoard')
    _ = input()
except KeyboardInterrupt:
    writer.close()
    