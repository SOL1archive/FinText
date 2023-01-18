import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from Data.Dataset import FinTextDataset
from Data.DataLoader import FinTextDataLoader
from Model.MainModel import FinTextModel

class TrainingApp:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.writer = SummaryWriter(log_dir='./runs')

    def prepare_dataset(self):
        self.dataset = FinTextDataset()
        self.dataloader = FinTextDataLoader(self.dataset)

    def prepare_model(self):
        self.model = FinTextModel().to(self.device)

    def train(self):
        criterion = nn.CrossEntropyLoss().to(self.device)
        optimizer = Adam(self.model.parameters())

        for epoch in range(self.dataloader):
            for i, (input_tensor, labels) in enumerate(self.dataloader):
                # Forward
                input_tensor = input_tensor.to(self.device)
                output_tensor = self.model(input_tensor)
                
                loss = criterion(output_tensor, labels)
                self.writer.add_scalar()

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.writer.flush()

    def main(self):
        self.prepare_dataset()
        self.prepare_model()
        self.train()

        try:
            print('Press Ctrl + C to turn off the TensorBoard')
            _ = input()
        except KeyboardInterrupt:
            self.writer.close()

if __name__ == '__main__':
    TrainingApp().main()
