import datetime
import os
from pprint import pprint

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from Data.Dataset import FinTextDataset, concat_dataset, to
from Data.DataLoader import FinTextDataLoader
from Model.MainModel import FinTextModel

class TrainTestApp:
    def __init__(self, **config):
        default_config = {
            "num_epoch": 30,
            "lr": 0.001,
            'lr_scheduler': None,
            'train_size': 0.8
        }

        for key in default_config.keys():
            if key not in config.keys():
                config[key] = default_config[key]

        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'max_split_size_mb:128'

        self.config = config
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
        print('Preparing datasets')
        self.prepare_dataset()
        self.prepare_dataloader()

    def prepare_dataset(self):
        def idx2path(idx):
            return f'./execution/dataset/dataset{idx}.dataset'
        dataset_lt = []
        for idx in range(3):
            dataset_lt.append(FinTextDataset(idx2path(idx)))

        self.dataset = concat_dataset(dataset_lt)
        
        #만약 전체 데이터셋을 GPU device에 올릴 수 있는 경우 다음 주석 해제
        #self.dataset.to(self.device)
        self.train_dataset, self.test_dataset = self.dataset.train_test_split(self.train_size)
        torch.cuda.empty_cache()

    def prepare_dataloader(self):
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=1)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=1)

    def prepare_model(self):
        self.model = FinTextModel()
        if self.use_cuda:
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
        criterion = nn.CrossEntropyLoss().to(self.device)

        print('Training')
        print(len(self.dataset))
        for epoch in range(self.num_epoch):
            for i, (inputs, labels) in enumerate(self.train_dataloader):
                print('training:', i)
                self.model.train()
                # Forward
                inputs = to(inputs, self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                self.train_writer.add_scalar("Loss/train", loss.item(), epoch)

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Memory Detach
                loss.detach()
                del loss
                del inputs
                del outputs
                torch.cuda.empty_cache()

            self.scheduler.step()

            msg = f"EPOCH: {epoch}, Training Loss {loss.item():.5f}"
            print(msg)

            #validation/test
            with torch.no_grad():
                self.model.eval()
                total_test_loss = 0.
                total_cnt = 0.
                positive_cnt = 0.
                accurate_pred = 0.
                for i, (inputs, labels) in enumerate(self.test_dataloader):
                    inputs = to(inputs, self.device)
                    outputs = self.model(inputs)
                    test_loss = criterion(outputs, labels)
                    total_test_loss += float(test_loss)
                    
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
                
                loss.detach()
                del test_loss
                del total_test_loss
                del inputs
                del outputs
                torch.cuda.empty_cache()

        self.train_writer.flush()
        self.test_writer.flush()

    def save_model(self):
        torch.save(
                self.model,
                f"./model-dir/{datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')}.model",
            )

    def main(self):
        self.prepare_model()
        #self.prepare_tensorboard_writer()
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
