import os
import torch
import copy
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image as Image
import PIL, random, csv
import numpy as np
from augment import RandomHorizontalFlip, RandomErasing, RandomResizedCrop, ToTensor, ColorJitter, Normalize
import warnings, sys, cv2, math
warnings.filterwarnings("ignore")
import torch.distributed as dist
from utils import generate_target


class RAF_DB(data.Dataset):
    # 已经有对齐的人脸
    def __init__(self, args, mode):
        super(RAF_DB, self).__init__()

        with open(args.label_path, 'r', encoding='utf-8') as f:
            labels = f.readlines()
        self.image_names = [label.strip().split(" ")[0] for label in labels if label.strip().split("_")[0] == mode]
        self.image_names = [name.strip().split(".")[0] + "_aligned.jpg" for name in self.image_names]
        self.image_names = [os.path.join(args.image_path, name) for name in self.image_names]
        self.labels = [int(label.strip().split(" ")[-1]) for label in labels if label.strip().split("_")[0] == mode]
        # change to the sequence of (anger, feat, disgust, happiness. sadness, surprise, neutral)
        label_2_label = {1: 5, 2: 2, 3: 1, 4: 3, 5: 4, 6: 0, 7: 6}
        # class distribution: [705, 717, 281, 4772, 1982, 1290, 2524]
        self.labels = [label_2_label[label] for label in self.labels]

        self.img_size = args.img_size
        self.mode = mode
        self.args = args

        # pre-detected facial landmarks using HRNet
        self.landmarks = torch.load(f"./Model_Data/raf_{mode}_19_landmarks.tensor")

        mean = (0.5412, 0.4324, 0.3795)
        std = (0.2485, 0.2182, 0.2066) # calculated based on MS-Celeb 1M
        if mode == "train":
            self.augment = transforms.Compose([
                RandomHorizontalFlip(p=0.5),
                RandomResizedCrop(size=(args.img_size, args.img_size), scale=(0.75, 1)),
                ColorJitter(0.4, 0.4, 0.4, 0),
                ToTensor(),
                Normalize(mean, std),
                RandomErasing(p=0.2, scale=(0.02, 0.20))
            ])
            self._add_noise(args.noise_ratio) # symmetric noise
            # self._add_class_dependent_noise(args.noise_ratio) # asymmetric noise
        else:
            self.augment = transforms.Compose([
                ToTensor(),
                Normalize(mean, std)
            ])

    def __getitem__(self, item):
        image = Image.open(self.image_names[item]).convert("RGB").resize((self.img_size, self.img_size))
        augment_image, augment_landmark = self.augment([image, self.landmarks[item]])
        # heatmap for landmark detection (AwingLoss for landmark detection is a little bit better than L2 Loss)
        heatmap = np.zeros((self.args.landmark_num, self.img_size // 4, self.img_size // 4), dtype=np.float32)
        weight_map = np.zeros((self.args.landmark_num, self.img_size // 4, self.img_size // 4), dtype=np.float32)
        for i in range(heatmap.shape[0]):
            heatmap[i], weight_map[i] = generate_target(heatmap[i], (augment_landmark[i] / 4 + 0.5).int())
        # landmarks are actually not used in inference
        return augment_image, heatmap, weight_map, self.labels[item], item

    def __len__(self):
        return len(self.labels)

    def _add_noise(self, r):
        for i in range(len(self.labels)):
            if random.random() < r:
                shift = random.randint(1, 6)
                self.labels[i] = int((self.labels[i] + shift) % self.args.class_num)

    def _add_class_dependent_noise(self, r):
        # anger --> disgust
        # disgust --> anger
        # fear --> surprise
        # happiness --> neutral
        # sadness --> neutral
        # surprise --> anger
        # neutral --> sadness
        trans = {0: 1, 1: 0, 2: 5, 3: 6, 4: 6, 5: 0, 6: 4}
        for i in range(len(self.labels)):
            if random.random() < r:
                self.labels[i] = trans[self.labels[i]]

# progressively balanced sampling
class MySampler(data.Sampler):
    def __init__(self, dataset, args, num_replicas=None, rank=None, shuffle=True):
        super(MySampler, self).__init__(dataset)
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

        self.args = args
        self.num = self.get_num()
        self.idx = self.get_idx()

    def __iter__(self):
        current_num = self.set_num()
        current_idx = self.set_idx(current_num)
        current_idx = torch.tensor(current_idx, dtype=torch.long)

        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = current_idx[torch.randperm(len(self.dataset), generator=g)].tolist()
        else:
            indices = current_idx[list(range(len(self.dataset)))].tolist()

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

    def get_num(self):
        num = [0 for i in range(self.args.class_num)]
        for i in range(len(self.dataset)):
            num[self.dataset.labels[i]] += 1
        return num

    def set_num(self):
        prob = [item / len(self.dataset) for item in self.num]
        prob = [max(1 - self.epoch / self.args.max_epoch, 0) * item + min(self.epoch / self.args.max_epoch, 1) * (
                    1 / self.args.class_num) for item in prob]
        num = [math.ceil(prob[i] * len(self.dataset)) for i in range(len(prob) - 1)]
        num.append(len(self.dataset) - sum(num))
        return num

    def get_idx(self):
        idx = [[] for i in range(self.args.class_num)]
        for i in range(len(self.dataset)):
            idx[self.dataset.labels[i]].append(i)
        return idx

    def set_idx(self, num):
        idx_all = []
        np.random.seed(self.epoch)
        for i in range(self.args.class_num):
            if num[i] <= self.num[i]:
                idx_all += np.random.choice(self.idx[i], num[i], replace=False).tolist()
            else:
                for j in range(num[i] // self.num[i]):
                    idx_all += self.idx[i]
                idx_all += np.random.choice(self.idx[i], num[i] % self.num[i], replace=False).tolist()
        return idx_all


if __name__ == "__main__":
    pass