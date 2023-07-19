import torch
import torch.nn as nn
import torch.nn.functional as F
import math, cv2
import numpy as np
import torch.distributed as dist

# generate target heatmap for landmark detection
def generate_target(img, pt, sigma=1.5, label_type='Gaussian'):
    # Check that any part of the gaussian is in-bounds
    tmp_size = sigma * 3
    ul = [int(pt[0] - tmp_size), int(pt[1] - tmp_size)]
    br = [int(pt[0] + tmp_size + 1), int(pt[1] + tmp_size + 1)]
    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return img, np.ones(shape=img.shape, dtype=np.float32)

    # Generate gaussian
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if label_type == 'Gaussian':
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    else:
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5)

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    dilate = cv2.dilate(img, np.ones((3, 3), dtype=np.uint8))
    WeightMap = np.ones_like(img)
    WeightMap[np.where(dilate > 0.2)] = 10

    return img, WeightMap

# Adaptive Wing Loss for Robust Face Alignment via Heatmap Regression (https://arxiv.org/abs/1904.07399)
class AWingLoss(object):
    def __init__(self, alpha=2.1, omega=14., epsilon=1., theta=0.5):
        self.alpha = alpha
        self.omega = omega
        self.epsilon = epsilon
        self.theta = theta

    def __call__(self, y_pred, y):
        lossMat = torch.zeros_like(y_pred)
        A = self.omega * (1 / (1 + (self.theta / self.epsilon) ** (self.alpha - y))) * (self.alpha - y) * (
                    (self.theta / self.epsilon) ** (self.alpha - y - 1)) / self.epsilon
        C = self.theta * A - self.omega * torch.log(1 + (self.theta / self.epsilon) ** (self.alpha - y))
        case1_ind = torch.abs(y - y_pred) < self.theta
        case2_ind = torch.abs(y - y_pred) >= self.theta
        lossMat[case1_ind] = self.omega * torch.log(
            1 + torch.abs((y[case1_ind] - y_pred[case1_ind]) / self.epsilon) ** (self.alpha - y[case1_ind]))
        lossMat[case2_ind] = A[case2_ind] * torch.abs(y[case2_ind] - y_pred[case2_ind]) - C[case2_ind]

        return lossMat


class Scheduler(object):
    def __init__(self):
        raise NotImplementedError

    def get_lr(self):
        raise NotImplementedError

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

class LinearWarmupLR(Scheduler):
    def __init__(self,
                 optimizer,
                 batches: int,
                 epochs: int,
                 base_lr: float,
                 target_lr: float = 0,
                 warmup_epochs: int = 0,
                 warmup_lr: float = 0,
                 last_iter: int = -1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(type(optimizer).__name__))
        self.optimizer = optimizer
        if last_iter == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
            last_iter = 0
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.baselr = base_lr
        self.learning_rate = base_lr
        self.total_iters = epochs * batches
        self.targetlr = target_lr
        self.total_warmup_iters = batches * warmup_epochs
        self.total_linear_iters = self.total_iters - self.total_warmup_iters
        self.total_lr_decay = self.baselr - self.targetlr
        self.warmup_lr = warmup_lr
        self.last_iter = last_iter
        self.step()

    def get_lr(self):
        if self.last_iter < self.total_warmup_iters:
            return self.warmup_lr + \
                (self.baselr - self.warmup_lr) * self.last_iter / self.total_warmup_iters
        else:
            linear_iter = self.last_iter - self.total_warmup_iters
            return self.baselr - self.total_lr_decay * linear_iter / self.total_linear_iters

    def step(self, iteration=None):
        """Update status of lr.

        Args:
            iteration(int, optional): now training iteration of all epochs.
                Usually no need to set it manually.
        """
        if iteration is None:
            iteration = self.last_iter + 1
        self.last_iter = iteration
        self.learning_rate = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rate

if __name__ == "__main__":
    pass