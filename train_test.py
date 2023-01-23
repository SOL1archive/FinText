import datetime
import logging

import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from Data.Dataset import FinTextDataset, concat_dataset
from Data.DataLoader import FinTextDataLoader
from Model.MainModel import FinTextModel

log = logging.getLogger(__name__)
stream_hander = logging.StreamHandler()
log.addHandler(stream_hander)

log_dir = __file__[:__file__.rfind('/')]
file_handler = logging.FileHandler(log_dir + '/log/train-test.log')
log.addHandler(file_handler)

# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

class TrainTestApp:
    def __init__(self, **config):
        default_config = {
            "num_epoch": 30,
            "lr": 0.001,
            'lr_scheduler': None,
            "df_list": [
                pd.read_pickle('./data-dir/data-df0.pkl')
            ],
            'train_size': 0.8
        }

        for key in default_config.keys():
            if key not in config.keys():
                config[key] = default_config[key]

        self.config = config
        self.df_list = self.config["df_list"]
        self.num_epoch = self.config["num_epoch"]
        self.lr = self.config['lr']
        self.train_size = self.config['train_size']
        if self.config['lr_scheduler'] is None:
            self.lr_lambda = lambda epoch: self.lr
        else:
            self.lr_lambda = self.config['lr_scheduler']

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.train_writer = None
        self.test_writer = None
        self.totalTrainingSamples_count = 0

        self.time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")

        self.prepare_dataset()
        self.prepare_dataloader()

    def prepare_dataset(self):
        def idx2path(idx):
            return f'execution/dataset/dataset{idx}.dataset'
        dataset_lt = []
        for idx in range(5):
            dataset_lt.append(FinTextDataset(idx2path(idx)))
        
        self.dataset = concat_dataset(dataset_lt)
        
        #만약 전체 데이터셋을 GPU device에 올릴 수 있는 경우 다음 주석 해제
        #self.dataset.to(self.device)
        self.train_dataset, self.test_dataset = self.dataset.train_test_split(self.train_size)

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
                log_dir=log_dir + 'train_cls'
            )
            self.test_writer = SummaryWriter(
                log_dir=log_dir + 'test_cls'
            )

    def prepare_optimizer(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.LambdaLR(
            optimizer=self.optimizer,
            lr_lambda=self.lr_lambda,
            last_epoch=-1,                            
            verbose=False
        )

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

            self.scheduler.step()

            msg = f"EPOCH: {epoch}, Training Loss {loss.item():.5f}"
            log.info(msg)
            if epoch % 10 == 0:
                print(msg)

            #validation/test
            with torch.no_grad():
                self.model.eval()
                total_test_loss = 0.
                total_cnt = 0.
                positive_cnt = 0.
                accurate_pred = 0.
                for i, (input_tensor, labels) in enumerate(self.test_dataloader):
                    outputs = self.model(input_tensor)
                    test_loss = self.loss(outputs, labels)
                    total_test_loss += test_loss.item()
                    
                    _, pred = torch.max(outputs.data, 1)
                    total_cnt += labels.size(0)
                    accurate_pred += (pred == labels).sum().item()
                    positive_cnt += (pred == 1).sum().item()
                    
                avg_test_loss = total_test_loss / len(self.test_dataloader)
                self.test_writer.add_scalar('avg loss/test', avg_test_loss, epoch)
                accuracy = 100 * accurate_pred / total_cnt
                precision = 100 * accurate_pred / positive_cnt
                self.test_writer.add_scalar('accuracy', accuracy, epoch)
                self.test_writer.add_scalar('precision', precision, epoch)

        log.info(f'End Training {datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")}')
        self.train_writer.flush()
        self.test_writer.flush()

    def save_model(self):
        torch.save(
                self.model,
                f"./model-dir/{datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')}.model",
            )

    def main(self):
        self.prepare_model()
        self.prepare_tensorboard_writer()
        self.prepare_optimizer()

        self.train()

        save_model = input("Save Model(Y/N)? ")
        if save_model.upper() == "Y":
            self.save_model()

        try:
            print("Press Ctrl + C to turn off the TensorBoard")
            _ = input()
        except KeyboardInterrupt:
            self.train_writer.close()
            self.test_writer.close()

if __name__ == "__main__":
    TrainTestApp().main()
