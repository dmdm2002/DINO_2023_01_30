import tqdm
import torch
from utils.Error_Loger import error_logger
from lightly.models.utils import update_momentum
# from .Trainer import TrainRunner
from utils.Option import Param


class One_Epoch(Param):
    def __init__(self):
        super(One_Epoch, self).__init__()

    def run_one_epoch_training(self, loader, ep, transform_list, model, criterion, optimizer):
        total_loss = 0
        for idx, (views, label, name) in enumerate(tqdm.tqdm(loader, desc=f'Now Epoch = [{ep}/{self.EPOCH}]')):
            update_momentum(model.student_backbone, model.teacher_backbone, m=0.99)
            update_momentum(model.student_head, model.teacher_head, m=0.99)
            views = [view.to(self.DEVICE) for view in views]

            label = label.to(self.DEVICE)

            global_views = views[:2]

            teacher_output = [model.forward_teacher(view) for view in global_views]
            student_output = [model.forward(view) for view in views]
            loss = criterion(teacher_output, student_output, epoch=ep)

            total_loss += loss.detach()

            loss.backward()
            model.student_head.cancel_last_layer_gradients(current_epoch=ep)
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = total_loss / len(loader)

        return model, avg_loss

    def run_one_epoch_test(self, loader, ep, transform_list, model, criterion, optimizer):
        total_loss = 0
        for idx, (item, label, name) in enumerate(tqdm.tqdm(loader, desc=f'Now Epoch = [{ep}/{self.EPOCH}]')):
            output = model.forward(item)
        avg_loss = total_loss / len(loader)

        return avg_loss

    def __call__(self, loader, ep, transform_list, model, criterion, optimizer):
        return self.run_one_epoch_training(loader, ep, transform_list, model, criterion, optimizer)