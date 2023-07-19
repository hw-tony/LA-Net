import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class score(nn.Module):
    def __init__(self, dim):
        super(score, self).__init__()
        self.fc_in = nn.Linear(dim, dim)
        self.fc_out = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, 2),
            nn.Softmax(dim=-1)
        )
    def forward(self, features):
        B, C = features.size()
        features = self.fc_in(features)
        f1, f2 = features[:, :C // 2], features[:, C // 2:]
        f = torch.cat([f1.unsqueeze(1).repeat(1, B, 1), f2.unsqueeze(0).repeat(B, 1, 1)], dim=-1)
        return self.fc_out(f)

class Label_Distribution_Estimation(nn.Module):
    # K neighbors in expression space & K neighbors in landmark space
    # neighborhood aggregation with learnable weights
    def __init__(self, K, labels, expansion, momentum):
        super(Label_Distribution_Estimation, self).__init__()
        self.K = K  # number of neighbors in fer space & landmark space
        # the EMA target starts with the one-hot label
        self.bank = F.one_hot(torch.tensor(labels, dtype=torch.long), num_classes=max(labels) + 1).float()
        self.fer_score = score(512 * expansion)
        self.lm_score = score(64 * expansion)
        self.momentum = momentum

    def forward(self, fer_features, lm_features, logits, idx):
        fer_neighbors = self._neighbor(fer_features)
        lm_neighbors = self._neighbor(lm_features)

        fer_weight = self.fer_score(fer_features)[:, :, 0] * fer_neighbors
        lm_weight = self.lm_score(lm_features)[:, :, 0] * lm_neighbors

        eps = 1e-8
        fer_weight = (fer_weight + eps / fer_neighbors.size(-1)) / (fer_weight.sum(-1, keepdim=True) + eps)
        lm_weight = (lm_weight + eps / lm_neighbors.size(-1)) / (lm_weight.sum(-1, keepdim=True) + eps)

        # target calculated in the current epoch
        fer_dis = fer_weight @ F.softmax(logits, dim=-1)
        lm_dis = lm_weight @ F.softmax(logits, dim=-1)
        targets = 0.5 * (fer_dis + lm_dis)
        return_targets = self.bank[idx].to(targets.device) * self.momentum + targets * (1 - self.momentum)
        self._update(targets.cpu().detach(), idx)
        return return_targets

    @torch.no_grad()
    def _neighbor(self, features):
        B, _ = features.size()
        similarity = F.normalize(features, dim=-1) @ F.normalize(features, dim=-1).T
        # exclude self-loop
        similarity[range(B), range(B)] = -1.

        top_idx = torch.topk(similarity, dim=-1, k=self.K, largest=True)[1]
        neighbor = torch.zeros_like(similarity, dtype=similarity.dtype, device=similarity.device)
        neighbor[torch.arange(B).unsqueeze(-1).repeat(1, self.K).view(-1), top_idx.view(-1)] = 1.0

        return neighbor

    @torch.no_grad()
    def _update(self, targets, idx):
        self.bank[idx] = self.bank[idx] * self.momentum + targets * (1 - self.momentum)