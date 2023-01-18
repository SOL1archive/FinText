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

METRICS_LABEL_NDX = 0
METRICS_PRED_NDX = 1
METRICS_LOSS_NDX = 2
METRICS_SIZE = 3


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

        self.trn_writer = None
        self.val_writer = None
        self.totalTrainingSamples_count = 0

        self.time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")

    def prepare_dataset(self):
        self.dataset = concat_dataset(self.df_list)
        self.dataset.to(self.device)

        self.dataloader = FinTextDataLoader(self.dataset)

    def prepare_model(self):
        self.model = FinTextModel()
        if self.use_cuda:
            log.info("Using CUDA; {} devices.".format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                self.model = nn.DataParallel(self.model)
            self.model = self.model.to(self.device)

    def prepare_optimizer(self):
        self.optimizer = Adam(self.model.parameters(), lr=self.config["lr"])

    def train(self):
        criterion = nn.CrossEntropyLoss().to(self.device)

        for epoch in range(self.num_epoch):
            for i, (input_tensor, labels) in enumerate(self.dataloader):
                # Forward
                input_tensor = input_tensor.to(self.device)
                output_tensor = self.model(input_tensor)

                loss = criterion(output_tensor, labels)
                self.writer.add_scalar()

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print(f"EPOCH: {epoch}, Training Loss {loss.item():.4f}")

        self.writer.flush()

    def main(self):
        self.prepare_dataset()
        self.prepare_model()
        self.prepare_optimizer()
        self.train()

        save_model = input("Save Model(Y/N)? ")
        if save_model.upper() == "Y":
            torch.save(
                self.model,
                f"./model-dir/{self.datetime.strftime('%d_%b_%Y_%H:%M:%S')}.model",
            )

        try:
            print("Press Ctrl + C to turn off the TensorBoard")
            _ = input()
        except KeyboardInterrupt:
            self.writer.close()

if __name__ == "__main__":
    TrainingApp().main()
