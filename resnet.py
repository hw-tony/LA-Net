import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from layers import BasicBlock, Bottleneck, conv1x1, conv3x3, Label_Distribution_Estimation
import torch.distributed as dist


class Landmark_ResNet(nn.Module):
    def __init__(self, args, block, layers, labels):
        super(Landmark_ResNet, self).__init__()

        self._norm_layer = nn.BatchNorm2d
        self.inplanes = 64
        self.dilation = 1
        replace_stride_with_dilation = [False, False, False]
        self.groups = 1
        self.base_width = 64

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.latlayer1 = conv1x1(64 * block.expansion, 64 * block.expansion)

        self.fer_layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.inplanes = 64 * block.expansion
        self.lm_layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.latlayer2 = conv1x1(128 * block.expansion, 64 * block.expansion)

        self.fer_layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.inplanes = 128 * block.expansion
        self.lm_layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.latlayer3 = conv1x1(256 * block.expansion, 64 * block.expansion)

        self.fer_layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.inplanes = 256 * block.expansion
        self.lm_layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.latlayer4 = conv1x1(512 * block.expansion, 64 * block.expansion)

        self.GAP = nn.AdaptiveAvgPool2d((1, 1))
        self.fer_fc = nn.Linear(512 * block.expansion, args.class_num)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.lm_fc = conv1x1(64 * block.expansion, args.landmark_num)

        self.smooth = Label_Distribution_Estimation(args.K, labels, block.expansion, args.dis_momentum)

        self.args = args
        # load from pretrained resnet-18/50
        self.load_parameters()

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def extract_base(self, images):
        features = self.conv1(images)
        features = self.bn1(features)
        features = self.relu(features)
        features = self.maxpool(features)
        features = self.layer1(features)
        return features

    def extract_fer(self, features):
        features = self.fer_layer2(features)
        features = self.fer_layer3(features)
        features = self.fer_layer4(features)
        features = torch.flatten(self.GAP(features), 1)
        return features

    def extract_lm(self, features):
        lm_features2 = self.lm_layer2(features)
        lm_features3 = self.lm_layer3(lm_features2)
        lm_features4 = self.lm_layer4(lm_features3)

        lm_features = self.latlayer4(lm_features4)
        lm_features = self.upsample(lm_features) + self.latlayer3(lm_features3)
        lm_features = self.upsample(lm_features) + self.latlayer2(lm_features2)
        lm_features = self.upsample(lm_features) + self.latlayer1(features)

        return lm_features

    def forward(self, images, idx):
        B = images.size(0)

        if not self.training:
            features = self.extract_base(images)
            features = self.extract_fer(features)
            return self.fer_fc(features)

        # extract expression and landmark features & perform prediction
        features = self.extract_base(images)
        fer_features = self.extract_fer(features)
        lm_features = self.extract_lm(features)
        fer_preds = self.fer_fc(fer_features)
        lm_preds = self.lm_fc(lm_features)

        # generate targets of each sample
        pooled_lm_features = torch.flatten(self.GAP(lm_features), 1)
        gathered_lm_features = gather_with_grad(pooled_lm_features)
        gathered_fer_features = gather_with_grad(fer_features)
        gathered_fer_preds = gather_with_grad(fer_preds)
        gathered_idx = concat_all_gather(idx).cpu()
        targets = self.smooth(gathered_fer_features,
                              gathered_lm_features,
                              gathered_fer_preds,
                              gathered_idx)[dist.get_rank() * B: (dist.get_rank() + 1) * B]

        return fer_preds, lm_preds, targets, fer_features, pooled_lm_features

    def load_parameters(self):
        assert self.args.pretrain_path != None
        parameters = torch.load(self.args.pretrain_path, map_location=torch.device(f"cuda:{self.args.local_rank}"))
        new_parameters = {}
        for name, p in parameters.items():
            if name.split(".")[0] in ["conv1", "bn1", "layer1"]:
                new_parameters[name] = p
            elif name.split(".")[0] == "criterion":
                continue
            elif name.split(".")[0] in ["layer2", "layer3", "layer4"]:
                new_parameters["fer_" + name] = p
                new_parameters["lm_" + name] = p
        self.load_state_dict(new_parameters, strict=False)

class MoCo_ResNet(nn.Module):
    def __init__(self, args, block, layer, labels):
        super(MoCo_ResNet, self).__init__()
        self.f_q = Landmark_ResNet(args, block, layer, labels)
        # momentum encoder for EL Loss
        self.f_k = Landmark_ResNet(args, block, layer, labels)
        self.fer_fc = nn.ModuleList([nn.Sequential(
            nn.Linear(512 * block.expansion, block.expansion * 64),
            nn.ReLU(inplace=True),
            nn.Linear(block.expansion * 64, block.expansion * 64)
        ) for _ in range(2)])
        self.lm_fc = nn.ModuleList([nn.Sequential(
            nn.Linear(64 * block.expansion, block.expansion * 64),
            nn.ReLU(inplace=True),
            nn.Linear(block.expansion * 64, block.expansion * 64)
        ) for _ in range(2)])
        self._initialize(self.f_q, self.f_k)
        self._initialize(self.fer_fc[0], self.fer_fc[1])
        self._initialize(self.lm_fc[0], self.lm_fc[1])

        self.register_buffer("fer_bank", torch.randn(block.expansion * 64, args.banksize))
        self.fer_bank = F.normalize(self.fer_bank, dim=0)
        self.register_buffer("lm_bank", torch.randn(block.expansion * 64, args.banksize))
        self.lm_bank = F.normalize(self.lm_bank, dim=0)
        self.register_buffer("target_bank", torch.empty(size=(args.banksize,)))
        self.target_bank.fill_(-1e10)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.args = args

    def forward(self, images, idx):
        if not self.training:
            return self.f_q(images, idx)
        fer_preds, lm_preds, targets_q, fer_q, lm_q = self.f_q(images, idx)
        fer_q = F.normalize(self.fer_fc[0](fer_q), dim=-1)
        lm_q = F.normalize(self.lm_fc[0](lm_q), dim=-1)

        with torch.no_grad():
            self._update(self.f_q, self.f_k)
            self._update(self.fer_fc[0], self.fer_fc[1])
            self._update(self.lm_fc[0], self.lm_fc[1])

            im_k, idx_this, idx_unshuffle = self._batch_shuffle_ddp(images)
            idx_k = concat_all_gather(idx)[idx_this]
            _, _, _, fer_k, lm_k = self.f_k(im_k, idx_k)
            fer_k = F.normalize(self.fer_fc[1](fer_k), dim=-1)
            lm_k = F.normalize(self.lm_fc[1](lm_k), dim=-1)
            fer_k = self._batch_unshuffle_ddp(fer_k, idx_unshuffle)
            lm_k = self._batch_unshuffle_ddp(lm_k, idx_unshuffle)

        confidence_q, pseudo_q = targets_q.max(1)
        low_q_idx = confidence_q < self.args.confidence
        pseudo_q[low_q_idx] = -idx[low_q_idx]

        loss_1 = self._cal(fer_q, lm_k, self.lm_bank.clone().detach(), pseudo_q)
        loss_2 = self._cal(lm_q, fer_k, self.fer_bank.clone().detach(), pseudo_q)
        cl_loss = loss_1 + loss_2

        self._dequeue_and_enqueue(fer_k, lm_k, pseudo_q)

        return fer_preds, lm_preds, targets_q, cl_loss

    def _cal(self, q, k, bank, label):
        N = q.size(0)
        n_mask = (label.unsqueeze(1) == label.unsqueeze(0)).float()
        d_mask = (label.unsqueeze(1) == self.target_bank.unsqueeze(0)).float()

        pos = (q @ k.T).reshape(N * N, 1)
        neg = (q @ bank + d_mask * -1e10).unsqueeze(1).repeat(1, N, 1).reshape(N * N, -1)
        cl_labels = torch.zeros((N * N,), dtype=torch.long, device=q.device)
        loss = nn.CrossEntropyLoss(reduction='none')(torch.cat([pos, neg], dim=-1) / self.args.con_T,
                                                     cl_labels).reshape(N, N)
        loss = (loss * n_mask / n_mask.sum(-1, keepdim=True)).sum(-1).mean()
        return loss

    @torch.no_grad()
    def _dequeue_and_enqueue(self, fer_k, lm_k, pseudo):
        fer_keys = concat_all_gather(fer_k)
        lm_keys = concat_all_gather(lm_k)
        pseudo_keys = concat_all_gather(pseudo)
        batch_size = fer_keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.args.banksize % batch_size == 0

        self.fer_bank[:, ptr:ptr + batch_size] = fer_keys.T
        self.lm_bank[:, ptr:ptr + batch_size] = lm_keys.T
        self.target_bank[ptr:ptr + batch_size] = pseudo_keys
        ptr = (ptr + batch_size) % self.args.banksize
        self.queue_ptr[0] = ptr

    def _update(self, q, k):
        for param_q, param_k in zip(q.parameters(), k.parameters()):
            param_k.copy_(param_k * self.args.moco_momentum + param_q.detach() * (1 - self.args.moco_momentum))
        for buffer_q, buffer_k in zip(q.buffers(), k.buffers()):
            buffer_k.copy_(buffer_q)

    def _initialize(self, q, k):
        for param_k in k.parameters():
            param_k.requires_grad = False
        k.load_state_dict(q.state_dict())

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_this, idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]


@torch.no_grad()
def concat_all_gather(tensor):
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class GatherLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[torch.distributed.get_rank()]
        return grad_out


def gather_with_grad(tensor):
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    tensors_gather = GatherLayer.apply(tensor)
    output = torch.cat(tensors_gather, dim=0)
    return output


if __name__ == "__main__":
    pass