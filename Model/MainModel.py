import torch
import torch.nn as nn
import torch.functional as F

from transformers import ElectraTokenizer, ElectraModel

from utils import CNN_Layer

class FinTextModel(nn.Module):
    def __init__(self, hyper_parameters=None):
        super().__init__()

        if hyper_parameters == None:
            self.hyper_parameters = {
                'community': {
                    'cnn_out_channels': 10,
                    'kernel_size': (9, 3),
                    'stride': 1,
                    'same': True
                },
                'article': {
                    'cnn_out_channels': 10,
                    'kernel_size': (9, 3),
                    'stride': 1,
                    'same': True
                }
            }
        else:
            self.hyper_parameters = hyper_parameters

        self.ko_tokenizer = ElectraTokenizer.from_pretrained('monologg/koelectra-base-v3-discriminator')
        self.ko_model = ElectraModel.from_pretrained('monologg/koelectra-base-v3-discriminator')
        self.kc_tokenizer = ElectraTokenizer.from_pretrained('beomi/KcELECTRA-base-v2022')
        self.kc_model = ElectraModel.from_pretrained('beomi/KcELECTRA-base-v2022')

        self.community_cnn = CNN_Layer(
            in_channels=1,
            out_channels=self.hyper_parameters['community']['cnn_out_channels'],
            kernel_size=self.hyper_parameters['community']['kernel_size'],
            stride=self.hyper_parameters['community']['stride'],
            padding=(
                self.hyper_parameters['community']['kernel_size'][0] // 2,
                self.hyper_parameters['community']['kernel_size'][1] // 2,
                ) if self.hyper_parameters['community']['same'] else 0
        )

        self.article_cnn = CNN_Layer(
            in_channels=1,
            out_channels=self.hyper_parameters['article']['cnn_out_channels'],
            kernel_size=self.hyper_parameters['article']['kernel_size'],
            stride=self.hyper_parameters['article']['stride'],
            padding=(
                self.hyper_parameters['article']['kernel_size'][0] // 2,
                self.hyper_parameters['article']['kernel_size'][1] // 2,
                ) if self.hyper_parameters['article']['same'] else 0
        )

    def forward():
        pass
