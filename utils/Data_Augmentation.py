import torch
import torchvision.transforms as transforms

from utils.Option import Param
from utils.Error_Loger import error_logger


class data_aug(Param):
    def __init__(self):
        """
        :var self.place: Now python file name and class
        """
        super(data_aug, self).__init__()
        self.place = 'data_aug class in Data_Augmentation (Global and Local Crop)'

    def global_transform(self):
        """
        RandomResizeCrop: 50%~100%로 crop 한다. self.INPUT_SIZE로 Resize까지 진행된다.
        :return transform: Global crop transform
        """
        transform = transforms.Compose([
            transforms.Resize(512),
            transforms.RandomResizedCrop((self.INPUT_SIZE, self.INPUT_SIZE), scale=(0.5, 1)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        return transform

    def local_transform(self):
        """
        RandomResizedCrop: 5% ~ 50%로 crop 한다. self.INPUT_SIZE로 Resize까지 진행
        :return: local crop transform
        """
        transform = transforms.Compose([
            transforms.Resize(512),
            transforms.RandomResizedCrop((self.INPUT_SIZE, self.INPUT_SIZE), scale=(0.05, 0.5)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        return transform

    def __call__(self, image, name):
        crops = []
        try:
            for _ in range(self.GLOBAL_CROP_NUMBERS):
                crops.append(self.global_transform())

            for _ in range(self.LOCAL_CROP_NUMBERS):
                crops.append(self.local_transform())

            return crops

        except Exception as e:
            error_logger(name, place=self.place, function='__call__', e=e)