from __future__ import absolute_import

import os
import numpy as np
from misc.dataset import ImageDataset
from misc.utils import data_parallel

import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable

_BATCH_SIZE = 16
_GPU_ID = 1

image_dir = "/home/liusheng/MSVD/Orig_images/"
feature_dir = "/home/liusheng/MSVD/Inception_features/"

inception_model = torchvision.models.inception_v3(pretrained=True)
# resnet_model = torchvision.models.resnet152(pretrained=True)
net = nn.Sequential(*list(inception_model.children())[:-1])
# net = nn.Sequential(*list(resnet_model.children())[:-1])
net = net.cuda(_GPU_ID)
print 'Inception v3\n', net

dataset = ImageDataset(image_dir,
                       feature_dir,
                       num_hierachy=1)
loader = data.DataLoader(dataset, batch_size=_BATCH_SIZE, collate_fn=dataset.collate_fn)

# it, fp = next(iter(loader))
# it = Variable(it.float()).cuda(_GPU_ID)
# print it.size()
# features = net(it)
# features = features.data.cpu().numpy()
# print features
# print features.shape

print 'Total number of images:', loader.batch_size * len(loader)
count = 1
for it, fp in loader:
    it = Variable(it.float()).cuda(_GPU_ID)
    features = net(it)
    features = np.squeeze(features.data.cpu().numpy())
    for ix, f in enumerate(fp):
        ft = features[ix, :]
        np.save(f, ft)
    if count % 100 == 0:
        print 'Processed %d (%.2f%%) images:' % (count*loader.batch_size, count / len(loader) * 100)
    count += 1