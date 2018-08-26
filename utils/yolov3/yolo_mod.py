from __future__ import division
import pickle as pkl
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2

from .models.darknet import Darknet, write_results
from utils import *
import os
from .data import load_classes, mark_box

DETIMAGE_DIM = 416
OBJECTNESS_THRESHOLD = 0.5
NONMAXSUPRESS_THRESHOLD = 0.4

def toggle_cv_PIL(x):
    return x[:,:,(2,1,0)]

# IMAGE SIZE HANDLING WHEN I/O
# TODO: Copy from example, to be consolidated with toolbox in the package
def letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w / img_w, h / img_h))
    new_h = int(img_h * min(w / img_w, h / img_h))
    resized_image = cv2.resize(img, (new_w, new_h),
                               interpolation=cv2.INTER_CUBIC)

    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h - new_h) // 2:(h - new_h) // 2 + new_h,
    (w - new_w) // 2:(w - new_w) // 2 + new_w, :] = resized_image

    return canvas


def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.
    img: opencv BGR image

    Returns a Variable
    """
    img = (letterbox_image(img, (inp_dim, inp_dim)))
    img = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img

################################################################################

class YoloMod:
    def __init__(self, global_args):
        """
        :param global_args: global setting of the service
        """
        self.args = global_args
        self.pwd_ = os.path.dirname(os.path.abspath(__file__))
        fpth = lambda p: os.path.join(self.pwd_, p)
        self.class_names = load_classes(fpth("data/coco.names"))
        self.dnet = Darknet(fpth("cfg/yolov3.cfg"), global_args.device)
        weight_file = fpth('checkpoints/yolov3.weights')
        self.dnet.load_net_binary(weight_file)
        with open(fpth("data/pallete"), "rb") as f:
            self.class_colors = pkl.load(f)
        self.dnet.eval()
        self.dnet.to(global_args.device)
        return

    def process(self, im):
        """
        :param im: a PIL image
        :type im: np.ndarray
        :return:
        """
        if im.ndim != 3:
            return im
        im_cv = toggle_cv_PIL(im)
        im = prep_image(im_cv, DETIMAGE_DIM).to(self.args.device)
        pred = self.dnet(im)
        print("pred")
        batch_det_boxes, batch_classes, batch_scores = \
            write_results(pred=pred,
                          confid_threshold=OBJECTNESS_THRESHOLD,
                          nms_conf=NONMAXSUPRESS_THRESHOLD)
        print("write_results")
        det_boxes = batch_det_boxes[0].cpu()
        classes = batch_classes[0].cpu()
        scores = batch_scores[0].cpu()

        outim = im_cv.copy()
        mark_box(outim, det_boxes, classes, scores,
                 self.class_names, self.class_colors, DETIMAGE_DIM)
        print("write_box")
        # write detection results
        # print(outim.shape)
        # cv2.putText(outim, "test", (20,20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0))
        outim = toggle_cv_PIL(outim)
        return outim

