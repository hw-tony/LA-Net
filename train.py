import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloader import RAF_DB, MySampler
from resnet import MoCo_ResNet
from layers import BasicBlock, Bottleneck
import time, random, argparse
import torch.distributed as dist
import numpy as np
from loss import MyLoss


def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train(args):

    train_dataset = RAF_DB(args, "train")
    test_dataset = RAF_DB(args, "test")

    train_sampler = MySampler(train_dataset, args)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size // dist.get_world_size(), pin_memory=True,
                              num_workers=4, sampler=train_sampler, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size // dist.get_world_size(), pin_memory=True,
                             num_workers=4)

    if args.model_type == "resnet-18":
        model = MoCo_ResNet(args, BasicBlock, [2, 2, 2, 2], train_dataset.labels).cuda()
    else:  # 'resnet-50'
        model = MoCo_ResNet(args, Bottleneck, [3, 4, 6, 3], train_dataset.labels).cuda()
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

    optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,
                                                  start_factor=1.0,
                                                  end_factor=0.,
                                                  total_iters=(len(train_dataset) // args.batch_size) * args.max_epoch)
    criterion = MyLoss(args)
    best_test_acc = 0

    for epoch in range(args.max_epoch):
        begin = time.time()

        train_acc, train_num = 0, 0
        criterion._reset()
        train_loader.sampler.set_epoch(epoch)

        model.train()
        for train_iter, (images, heatmap, weight_map, labels, idx) in enumerate(train_loader):
            optimizer.zero_grad()

            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            heatmap = heatmap.cuda(non_blocking=True)
            weight_map = weight_map.cuda(non_blocking=True)

            fer_preds, lm_preds, targets, cl_loss = model(images, idx)
            loss = criterion(fer_preds, labels, targets, lm_preds, heatmap, weight_map, cl_loss, idx)

            loss.backward()
            optimizer.step()
            scheduler.step()
            train_acc += (fer_preds.max(1)[1] == labels).sum().item()
            train_num += labels.size(0)

        test_acc, test_num = 0, 0
        model.eval()
        with torch.no_grad():
            for test_iter, (images, _, _, labels, _) in enumerate(test_loader):
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)

                fer_preds = model(images, None)

                test_acc += (fer_preds.max(1)[1] == labels).sum().item()
                test_num += labels.size(0)

        test_acc = test_acc / test_num
        best_test_acc = max(best_test_acc, test_acc)

        if args.local_rank == 0:
            print(f"epoch:{epoch} "
                  f"train_acc:{round(train_acc / train_num, 4)} "
                  f"test_acc:{round(test_acc, 4)} "
                  f"cost_time:{round(time.time() - begin, 4)}s", end=' ')
            criterion._print()

    if args.local_rank == 0:
        print(f"best_test_acc: {best_test_acc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0, help='node rank for distributed training')
    
    # path to datasets
    parser.add_argument("--image_path", type=str, default="./RAF_DB/basic/Image/aligned")
    parser.add_argument("--label_path", type=str, default="./RAF_DB/basic/EmoLabel/list_patition_label.txt")
    
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--landmark_img_size", type=int, default=256)
    parser.add_argument("--landmark_num", type=int, default=19)

    parser.add_argument("--max_epoch", type=int, default=80)
    parser.add_argument("--base_lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--batch_size", type=int, default=128)

    # weight for loss
    parser.add_argument("--lam_lm", type=float, default=1.)
    parser.add_argument("--lam_cl", type=float, default=0.1)
    parser.add_argument("--lam_kl", type=float, default=1.)

    # parameters for EL Loss
    parser.add_argument("--con_T", type=float, default=0.1)
    parser.add_argument("--confidence", type=float, default=0.7)
    parser.add_argument("--banksize", type=int, default=8192)
    parser.add_argument("--moco_momentum", type=float, default=0.999)

    # parameters for LDE
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--dis_momentum", type=float, default=0.9)

    # add synthetic noise
    parser.add_argument("--noise_ratio", type=float, default=0.)

    parser.add_argument("--class_num", type=int, default=7, help="number of classes")
    parser.add_argument("--model_type", type=str, choices=["resnet-18", "resnet-50"], default="resnet-18")
    
    # path to your pretrained resnet state_dict
    parser.add_argument("--pretrain_path", type=str, default='./Model_Data/resnet-18.pth' ,help="file path of pretrained model")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend="nccl")
    set_seed(args.seed)
    assert args.batch_size % dist.get_world_size() == 0

    train(args)
