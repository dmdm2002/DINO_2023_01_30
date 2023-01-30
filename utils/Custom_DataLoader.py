import sys
import traceback

import torch.utils.data as data
import PIL.Image as Image
import pandas as pd

from utils.Error_Loger import error_logger


class Loader(data.DataLoader):
    def __init__(self, root_path, run_type='train', transform=None):
        """
        DataLoader Custom DataSet

        :param root_path: DATABASE PATH
        :param run_type: Model run type (train, test, valid)
        :param transform: Image style transform module -> 디노에서는 training loop에서 적용하는게 좋을꺼 같다.

        :var self.data_info: Dataframe with DB information for us to use [folder, image name, class]
        :var self.path_list: Data Path information list
        :var self.label_list: Data class information list
        """
        super(Loader, self).__init__(self)
        self.place = 'Loader class in Custom_DataLoader'
        self.root_path = root_path
        self.run_type = run_type
        self.transform = transform

        data_info = None
        try:
            data_info = pd.read_csv(f'{self.root_path}/{self.run_type}.csv')
        except Exception as e:
            # print(traceback.format_exc())
            error_logger('None', self.place, '__init__', e)

        assert self.run_type == 'train' or self.run_type == 'test' or self.run_type == 'valid', \
            'Only train, test, and valid are available for run_type.'

        if self.run_type == 'test':
            self.path_list = self.get_paths(data_info)

        elif self.run_type == 'train' or self.run_type == 'valid':
            self.path_list = self.get_paths(data_info)
            self.label_list = self.get_labels(data_info)

    def get_paths(self, data_info):
        """
        Get Path information in Data information

        :param data_info: Dataframe with DB information for us to use [folder, image name, class]
        :var paths_list: full path information list
        :return: full path information
        """
        paths_list = []
        paths_info = data_info.iloc[:, 0:2].values

        return self.make_path_list(paths_info, paths_list)

    def make_path_list(self, path_info, paths_list: list):
        """
        Make Path list using Path information

        :param path_info: Extract path info only from data_info [path info: folder, image name]
        :param paths_list: Complete path information created from extracted information
        :return: List with complete path information
        """
        try:
            for folder, image_name in path_info:
                full_path = f'{self.root_path}/{self.run_type}/{folder}/{image_name}'
                paths_list.append(full_path)

            return paths_list

        except Exception as e:
            error_logger('None', self.place, 'make_path_list', e)

    def get_labels(self, data_info):
        """
        Get Label information in Data information

        :param data_info: Dataframe with DB information for us to use [folder, image name, class]
        :return: Extract class info only from data_info
        """
        return data_info.iloc[:, 2:].values

    def get_ImageName(self, path: str):
        """
        Get Image Name in Path information

        :param path: Full path to the image
        :return: Image Name
        """
        ImageName = path.split('/')[-1]
        ImageName = ImageName.split(".")[0]

        return ImageName

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index: int):
        try:
            if self.run_type == 'test':
                item = self.transform(Image.open(self.path_list[index]))
                image_name = self.get_ImageName(self.path_list[index])

                return [item, image_name]

            else:
                item = self.transform(Image.open(self.path_list[index]))
                label = self.label_list[index]
                image_name = self.get_ImageName(self.path_list[index])

                return [item, label, image_name]

        except Exception as e:
            image_name = self.get_ImageName(self.path_list[index])
            error_logger(image_name, place=self.place, function='__getitem__', e=e)