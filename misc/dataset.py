import os
import random
# import transforms
import numpy as np
from PIL import Image

import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable


class ImageDataset(data.Dataset):
    def __init__(self, image_dir, feature_dir,
                 transforms=None, num_hierachy=1):
        assert os.path.exists(image_dir)
        if not os.path.exists(feature_dir):
            os.makedirs(feature_dir)

        if not num_hierachy in [1, 2]:
            raise ValueError('Bad `num_hierachy`, must be one of 1, 2.')
        self.num_hierachy = num_hierachy
        self.images = []
        if self.num_hierachy == 1:
            folders = [(os.path.join(image_dir, i), os.path.join(feature_dir, i))
                       for i in sorted(os.listdir(image_dir))]
            for image_fd, feature_fd in folders:
                if not os.path.exists(feature_fd):
                    os.makedirs(feature_fd)
                self.images += [(os.path.join(image_fd, i), os.path.join(feature_fd, os.path.splitext(i)[0]+'.npy'))
                                for i in sorted(os.listdir(image_fd))]
        else:
            cls = os.listdir(image_dir)
            folders = [(os.path.join(image_dir, c, i), os.path.join(feature_dir, c, i))
                       for c in cls for i in os.listdir(os.path.join(image_dir, c))]
            for image_fd, feature_fd in folders:
                if not os.path.exists(feature_fd):
                    os.makedirs(feature_fd)
                self.images += [(os.path.join(image_fd, i), os.path.join(feature_fd, os.path.splitext(i)[0]+'.npy'))
                                for i in sorted(os.listdir(image_fd))]
        self.images = sorted(self.images, key=lambda x: x[0].lower())

        # self.trans = transforms
        self.trans = torchvision.transforms.Compose([
            torchvision.transforms.Resize((299, 299), interpolation=Image.BILINEAR),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[.485, .456, .406],
                std=[.229, .224, .225])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path, feature_path = self.images[index]
        image = Image.open(image_path)
        np_image = np.array(image)
        # print(np_image.mean(axis=(0, 1)))
        if self.trans is not None:
            image = self.trans(image)
        # print(image.mean())
        return image, feature_path

    def collate_fn(self, batch):
        images = [b[0][None, :] for b in batch]
        feature_paths = [b[1] for b in batch]
        images = torch.cat(images, dim=0)
        return images, feature_paths



if __name__ == "__main__":
    image_dir = "/home/liusheng/MSVD/Orig_images/"
    feature_dir = "/home/liusheng/MSVD/Inception_features/"
    dataset = ImageDataset(image_dir,
                           feature_dir,
                           num_hierachy=1)
    # it, fp = dataset[0]
    # print fp
    # print it
    loader = data.DataLoader(dataset, batch_size=2, collate_fn=dataset.collate_fn)
    it, fp = next(iter(loader))
    print fp
    print it