import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import AWingLoss
import math


class MyLoss(object):
    def __init__(self, args):
        self.loss_1 = 0
        self.loss_2 = 0
        self.loss_3 = 0
        self.loss_4 = 0

        self.args = args
        self.cnt = 0
        self.cur_epoch = 0
        self.awing = AWingLoss()

    def __call__(self, fer_preds, labels, targets, lm_preds, heatmap, weight_map, cl_loss, idx):
        self.cnt += 1

        # main loss
        loss_1 = F.cross_entropy(fer_preds, labels)
        self.loss_1 += loss_1.item()

        # landmark loss
        loss_2 = (self.awing(lm_preds, heatmap) * weight_map).mean()
        self.loss_2 += loss_2.item()

        # kl loss
        loss_3 = F.kl_div(F.log_softmax(fer_preds, dim=-1),
                          targets,
                          reduction='batchmean')
        self.loss_3 += loss_3.item()

        # contrastive loss
        self.loss_4 += cl_loss.item()

        loss = loss_1 + \
               self.args.lam_lm * loss_2 + \
               self.args.lam_kl * loss_3 + \
               self.args.lam_cl * cl_loss

        return loss

    def _reset(self):
        self.loss_1 = 0
        self.loss_2 = 0
        self.loss_3 = 0
        self.loss_4 = 0
        self.cnt = 0
        self.cur_epoch += 1

    def _print(self):
        print(f"fer loss:{round(self.loss_1 / self.cnt, 4)} "
              f"lm loss:{round(self.loss_2 / self.cnt, 4)} "
              f"kl loss:{round(self.loss_3 / self.cnt, 4)} "
              f"cl loss:{round(self.loss_4 / self.cnt, 4)}")