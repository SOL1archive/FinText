import datetime
import logging

import pandas as pd

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from Data.Dataset import FinTextDataset
from Data.DataLoader import FinTextDataLoader
from Model.MainModel import FinTextModel
from utils import concat_dataset

log = logging.getLogger(__name__)

# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

class TrainingApp:
    def __init__(self, **config):
        default_config = {
            "num_epoch": 30,
            "lr": 0.001,
            "df_list": [
                pd.read_csv("./data-dir/kakao.xlsx"),
                pd.read_csv("./data-dir/spc.xlsx"),
            ],
        }

        for key in default_config.keys():
            if key not in config.keys():
                config[key] = default_config[key]

        self.config = config
        self.df_list = self.config["df_list"]
        self.num_epoch = self.config["num_epoch"]

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.train_writer = None
        self.test_writer = None
        self.totalTrainingSamples_count = 0

        self.time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")

    def prepare_dataset(self):
        self.dataset = concat_dataset(self.df_list)
        self.dataset.to(self.device)
        self.train_dataset, self.test_dataset = self.dataset.train_test_split()

    def prepare_dataloader(self):
        self.train_dataloader = FinTextDataLoader(self.train_dataset)
        self.test_dataloader = FinTextDataLoader(self.test_dataset)

    def prepare_model(self):
        self.model = FinTextModel()
        if self.use_cuda:
            log.info(f"Using CUDA: {torch.cuda.device_count()} devices.")
            if torch.cuda.device_count() > 1:
                self.model = nn.DataParallel(self.model)
            self.model = self.model.to(self.device)

    def prepare_tensorboard_writer(self):
        if self.train_writer is None:
            log_dir = f'./runs/{self.time_str}/'

            self.train_writer = SummaryWriter(
                log_dir=log_dir + 'train_cls')
            self.test_writer = SummaryWriter(
                log_dir=log_dir + 'test_cls')

    def prepare_optimizer(self):
        self.optimizer = Adam(self.model.parameters(), lr=self.config["lr"])

    def train(self):
        log.info(f'Start Training: {self.time_str}')
        criterion = nn.CrossEntropyLoss().to(self.device)

        for epoch in range(self.num_epoch):
            for i, (input_tensor, labels) in enumerate(self.train_dataloader):
                self.model.train()
                # Forward
                input_tensor = input_tensor.to(self.device)
                output_tensor = self.model(input_tensor)

                loss = criterion(output_tensor, labels)
                self.train_writer.add_scalar("Loss/train", loss.item(), epoch)

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            msg = f"EPOCH: {epoch}, Training Loss {loss.item():.5f}"
            print(msg)
            log.info(msg)

            #validation/test
            with torch.no_grad():
                self.model.eval()
                total_test_loss = 0
                for i, (input_tensor, labels) in enumerate(self.test_dataloader):
                    outputs = self.model(input_tensor)
                    test_loss = self.loss(outputs, labels)
                    total_test_loss += test_loss.item()
                avg_test_loss = total_test_loss / len(self.test_dataloader)
                self.test_writer.add_scalar('avg loos/test', avg_test_loss, epoch)
                self.test_writer.add_scalar('total loss/test', total_test_loss, epoch)

        log.info(f'End Training {datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")}')
        self.train_writer.flush()
        self.test_writer.flush()

    def main(self):
        self.prepare_dataset()
        self.prepare_dataloader()
        self.prepare_model()
        self.prepare_tensorboard_writer()
        self.prepare_optimizer()

        self.train()

        save_model = input("Save Model(Y/N)? ")
        if save_model.upper() == "Y":
            torch.save(
                self.model,
                f"./model-dir/{datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')}.model",
            )

        try:
            print("Press Ctrl + C to turn off the TensorBoard")
            _ = input()
        except KeyboardInterrupt:
            self.train_writer.close()
            self.test_writer.close()

if __name__ == "__main__":
    TrainingApp().main()
