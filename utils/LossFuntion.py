import torch
import torch.nn.functional as F

from utils.Error_Loger import error_logger


class HLoss(object):
    def __init__(self, temperature_t: float, temperature_s: float):
        """
        :param temperature_t: teacher model output sharpening parameter
        :param temperature_s: student model output sharpening parameter
        """
        self.temperature_t = temperature_t
        self.temperature_s = temperature_s

    def __call__(self, t: torch.FloatTensor, s: torch.FloatTensor, center: torch.FloatTensor):
        try:
            t = F.softmax((t.detach() - center) / self.temperature_t, dim=1)
            log_s = F.log_softmax((s / self.temperature_s), dim=1)

            return -(t * log_s).sum(dim=1).mean()

        except Exception as e:
            error_logger('None', place='LossFunction', function='HLoss', e=e)


class CalLoss(object):
    def __init__(self, loss_fn, center):
        """
        :param loss_fn: Loss function
        :param center: Centering Parameter
        """
        self.loss_fn = loss_fn
        self.center = center

    def __call__(self, t1, t2, s1, s2):
        try:
            loss = self.loss_fn(t1, s2, self.center) + self.loss_fn(t2, s1, self.center)

            emperical_center = F.normalize(
                torch.cat([t1, t2]).mean(dim=0, keepdims=True),
                dim=-1,
            )

            return loss, emperical_center

        except Exception as e:
            error_logger('None', place='LossFunction', function='CalLoss', e=e)