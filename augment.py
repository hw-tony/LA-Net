import torch
import numpy as np
import torchvision.transforms as transforms
import random
from PIL import Image, ImageFilter
import math

class RandomHorizontalFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, input):
        image, landmark = input[0], input[1]
        if random.random() < self.p:
            h, w = image.size
            lm = landmark.clone()
            lm[:, 0] = h - lm[:, 0]
            return image.transpose(Image.FLIP_LEFT_RIGHT), lm
        return image, landmark

class Grayscale(object):
    def __init__(self, p):
        self.gray = transforms.RandomGrayscale(p)

    def __call__(self, input):
        image, landmark = input[0], input[1]
        return self.gray(image), landmark

class RandomResizedCrop(object):
    def __init__(self, size, scale, ratio=(3. / 4., 4. / 3.)):
        self.scale = scale
        self.ratio = ratio
        self.size = size

    def __call__(self, input):
        image, landmark = input[0], input[1]
        i, j, h, w = self.get_params(image, self.scale, self.ratio)
        img = image.crop((i, j, i + h, j + w))
        img = img.resize(self.size)
        lm = landmark - torch.tensor([[i, j]])
        lm = lm.float()
        lm[:, 0] = lm[:, 0] / h * self.size[0]
        lm[:, 1] = lm[:, 1] / w * self.size[1]
        lm = lm.int()
        return img, lm

    def get_params(self, image, scale, ratio):
        width, height = image.size
        area = height * width

        for _ in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if (in_ratio < min(ratio)):
            w = width
            h = int(round(w / min(ratio)))
        elif (in_ratio > max(ratio)):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

class ColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.color = transforms.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, input):
        image, landmark = input[0], input[1]
        return self.color(image), landmark

class ToTensor(object):
    def __init__(self):
        self.tensor = transforms.ToTensor()

    def __call__(self, input):
        image, landmark = input[0], input[1]
        return self.tensor(image), landmark

class RandomErasing(object):
    def __init__(self, p, scale, ratio=(0.3, 3.3)):
        self.erase = transforms.RandomErasing(p, scale, ratio)

    def __call__(self, input):
        image, landmark = input[0], input[1]
        return self.erase(image), landmark

class Normalize(object):
    def __init__(self, mean, std):
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def __call__(self, input):
        image, landmark = input[0], input[1]
        return self.normalize(image), landmark