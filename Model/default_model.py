import torch
import torch.nn as nn


class ResNet(nn.Module):
    def __init__(self, output_class):
        """
        This is a class that builds a default model
        :var self.model:
            ResNet34
        :param output_class:
            최종 우리가 원하는 class의 갯수
        """
        super(ResNet, self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
        # self.model.fc = nn.Linear(512, output_class, bias=True)

    def forward(self, x):
        """
        :param x :
            input image data
        :return :
            model(x) --> x가 거쳐서 나온 output feature map, size = (1 x 1 x 2)
        """
        return self.model(x)