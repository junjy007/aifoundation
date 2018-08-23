from __future__ import division
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2

from models.darknet import Darknet
from utils import *
import os
from data import load_classes, prep_image

test_dir = 'data/test_imgs'
class_names = load_classes("data/coco.names")
image_names = [os.path.join(test_dir, n_) for n_ in os.listdir(test_dir)]
images = list(map(cv2.imread, image_names))
input_images = [prep_image(im_, 416) for im_ in images]

dnet = Darknet("cfg/yolov3.cfg")
weight_file = 'checkpoints/yolov3.weights'
dnet.load_net_binary(weight_file)
dnet.eval()

#
l_0 = nn.Sequential(list(dnet.module_list)[0][:2])  # first conv layer
h = l_0(input_images[0])
h0 = torch.load(os.path.expanduser("~/tmp/c0a.pt"))
print((h-h0).abs().max())
print('')
#

test_range = list(range(80, 107))
det = dnet(input_images[0], test_range, os.path.expanduser("~/tmp/yolomy_"))


for i_ in test_range:
    mtype = dnet.mod_specs[i_]['type']
    fname_tut = os.path.expanduser("~/tmp/yolotut_") \
                + "{:03d}-{}.pt".format(i_, mtype)
    fname_my = os.path.expanduser("~/tmp/yolomy_") \
                + "{:03d}-{}.pt".format(i_, mtype)
    print(fname_tut, fname_my)
    res_tut = torch.load(fname_tut)
    res_my = torch.load(fname_my)

    print("Compare {}".format((res_tut-res_my).abs().max()))

torch.save(det, os.path.expanduser("~/tmp/myres.pt"))
print("Done")
