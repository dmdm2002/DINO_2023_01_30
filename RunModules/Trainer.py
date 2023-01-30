import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

from utils.Custom_DataLoader import Loader
from utils.Error_Loger import error_logger
from utils.Option import Param
from utils.Data_Augmentation import data_aug
from RunModules.Run_One_Epoch import One_Epoch

from lightly.data import DINOCollateFunction
from lightly.loss import DINOLoss
from Model.DINO import DINO_model

from torch.utils.tensorboard import SummaryWriter


class TrainRunner(Param):
    def __init__(self):
        super(TrainRunner, self).__init__()

        os.makedirs(self.OUTPUT_CKP, exist_ok=True)
        os.makedirs(self.OUTPUT_LOG, exist_ok=True)

        self.place = 'TrainRunner'

    def init_weight(self, module):
        class_name = module.__class__.__name__

        if class_name.find("Conv2d") != -1 and class_name.find("EdgeConv2d") == -1 and class_name.find("DynConv2d") == -1:
            nn.init.normal_(module.weight.data, 0.0, 0.02)

        elif class_name.find("BatchNorm2d") != -1:
            nn.init.normal_(module.weight.data, 1.0, 0.02)
            nn.init.constant(module.bias.data, 0.0)

    def set_transform(self):
        if self.AUG:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(self.INPUT_SIZE),
                # jiter나 rotate 같은 거 추가되면 좋을 듯. 근데 DINO Lib에서 자동으로 해주나..?
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(self.INPUT_SIZE),
            ])

        return transform

    def call_ckp(self, model, pretrain=False):
        if self.CKP_LOAD:
            print(f'Check Point [{self.LOAD_CKP_EPCOH}] Loading...')
            try:
                if pretrain:
                    epoch = 0

                else:
                    ckp = torch.load(f'{self.OUTPUT_CKP}/{self.LOAD_CKP_EPCOH}.pth')
                    model.load_state_dict(ckp['model_state_dict'])
                    epoch = ckp['epoch'] + 1

                return model, epoch

            except Exception as e:
                error_logger('None', self.place, 'run/CKP LOADING', e)
        else:
            print(f'Initialize Model Weight...')
            try:
                model.apply(self.init_weight)
                epoch = 0

                return model, epoch

            except Exception as e:
                error_logger('None', self.place, 'Initialize', e)

    def run(self):
        print('--------------------------------------')
        print(f'[RunType] : Training!!')
        print(f'[Device] : {self.DEVICE}!!')
        print('--------------------------------------')

        backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
        backbone, epoch = self.call_ckp(backbone, pretrain=True)
        backbone = nn.Sequential(*list(backbone.children())[:-1])
        # print(backbone)

        model = DINO_model(backbone, 512)
        model.to(self.DEVICE)

        # assert self.AUG is bool, 'Only boolean type is available for self.AUG.'
        data_transform = self.set_transform()
        tr_dataset = Loader(self.DATASET_PATH, run_type='train', transform=data_transform)
        # valid_dataset = Loader(self.DATASET_PATH, run_type='valid')

        transform_list = self.set_transform()

        collate_fn = DINOCollateFunction()

        criterion = DINOLoss(
            output_dim=2048,
            warmup_teacher_temp_epochs=5,
        )
        # move loss to correct device because it also contains parameters
        criterion = criterion.to(self.DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for ep in (epoch, self.EPOCH):
            model.train()
            tr_loader = DataLoader(dataset=tr_dataset, batch_size=self.BATCHSZ, collate_fn=collate_fn, shuffle=True)
            # valid_loader = DataLoader(dataset=valid_dataset, batch_size=self.BATCHSZ, collate_fn=collate_fn, shuffle=True)

            model, avg_loss = One_Epoch()(tr_loader, ep, transform_list, model, criterion, optimizer)
            print(f'Now Epoch [{ep}/{self.EPOCH}] --> Loss : {avg_loss:.5f}')
            # avg_loss = One_Epoch()