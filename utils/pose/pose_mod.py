# _*_coding:utf-8 _*_
# Author  : Tao
"""
This Code is ...
"""

import cv2
import math
import time
import os
import numpy as np
from .util import *
import torch
import torch.nn as nn
import argparse
from torch.autograd import Variable
from .pose_net import pose_model, contruct
from .config_reader import config_reader
from scipy.ndimage.filters import gaussian_filter

torch.set_num_threads(torch.get_num_threads())


class PoseEstimate():
    def __init__(self, global_args):
        """
        :param global_args: global setting of the service
        """
        self.args = global_args

    def process(self, im):
        """
        :param im:
        :return:
        """
        out_img = self.convert_img(im)

        return out_img

    def convert_img(self, oriImg):
        limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5],
                   [6, 7], [7, 8], [2, 9], [9, 10],
                   [10, 11], [2, 12], [12, 13], [13, 14],
                   [2, 1], [1, 15], [15, 17], [1, 16],
                   [16, 18], [3, 17], [6, 18]]

        mapIdx = [[31, 32], [39, 40], [33, 34], [35, 36],
                  [41, 42], [43, 44], [19, 20], [21, 22],
                  [23, 24], [25, 26], [27, 28], [29, 30],
                  [47, 48], [49, 50], [53, 54], [51, 52],
                  [55, 56], [37, 38], [45, 46]]

        # visualize
        colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0],
                  [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85],
                  [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255],
                  [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255],
                  [255, 0, 170], [255, 0, 85]]

        model = self.load_model()

        # param_, model_ = config_reader()

        tic = time.time()
        # oriImg = cv2.imread(test_image)  # B,G,R order

        print("cv_read", oriImg.shape)
        # ------------------------------------------------------------------

        with torch.no_grad():
            imageToTest = torch.transpose(torch.transpose(
                torch.unsqueeze(torch.from_numpy(oriImg).float(), 0),
                2, 3), 1, 2)

        print("ToTensor:", imageToTest.size())
        # ------------------------------------------------------------------

        multiplier = [x * 368 / oriImg.shape[0]
                      for x in [0.5, 1, 1.4, 1.8]]

        # print("xxxxx:", multiplier)

        heatmap_avg = torch.zeros((len(multiplier), 19,
                                   oriImg.shape[0],
                                   oriImg.shape[1])).to(self.args.device)
        paf_avg = torch.zeros((len(multiplier), 38,
                               oriImg.shape[0],
                               oriImg.shape[1])).to(self.args.device)

        toc = time.time()
        print('time is %.5f' % (toc - tic))
        tic = time.time()

        print("multiplier:", len(multiplier))
        # ------------------------------------------------------------------

        for m in range(len(multiplier)):
            scale = multiplier[m]

            print("scale: ", scale)

            h = int(oriImg.shape[0] * scale)
            w = int(oriImg.shape[1] * scale)
            pad_h = 0 if (h % 8 == 0) else 8 - (h % 8)
            pad_w = 0 if (w % 8 == 0) else 8 - (w % 8)
            new_h = h + pad_h
            new_w = w + pad_w

            imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale,
                                     interpolation=cv2.INTER_CUBIC)
            imageToTest_padded, pad = padRightDownCorner(imageToTest, 8, 128)
            imageToTest_padded = np.transpose(
                np.float32(imageToTest_padded[:, :, :, np.newaxis]),
                (3, 2, 0, 1)) / 256 - 0.5

            feed = torch.from_numpy(imageToTest_padded).to(self.args.device)
            output1, output2 = model(feed)

            heatmap = nn.functional.interpolate(
                output2,
                (oriImg.shape[0], oriImg.shape[1])).to(self.args.device)

            paf = nn.functional.interpolate(
                output1,
                (oriImg.shape[0], oriImg.shape[1])).to(self.args.device)

            heatmap_avg[m] = heatmap[0].data
            paf_avg[m] = paf[0].data

        toc = time.time()
        print('time is %.5f' % (toc - tic))
        tic = time.time()

        heatmap_avg = torch.transpose(
            torch.transpose(
                torch.squeeze(
                    torch.mean(heatmap_avg, 0)), 0, 1), 1, 2).to(self.args.device)
        paf_avg = torch.transpose(
            torch.transpose(
                torch.squeeze(
                    torch.mean(paf_avg, 0)), 0, 1), 1, 2).to(self.args.device)
        heatmap_avg = heatmap_avg.cpu().numpy()
        paf_avg = paf_avg.cpu().numpy()

        toc = time.time()
        print('time is %.5f' % (toc - tic))
        tic = time.time()

        all_peaks = []
        peak_counter = 0

        # maps =
        for part in range(18):
            map_ori = heatmap_avg[:, :, part]
            map = gaussian_filter(map_ori, sigma=3)
            map_left = np.zeros(map.shape)
            map_left[1:, :] = map[:-1, :]
            map_right = np.zeros(map.shape)
            map_right[:-1, :] = map[1:, :]
            map_up = np.zeros(map.shape)
            map_up[:, 1:] = map[:, :-1]
            map_down = np.zeros(map.shape)
            map_down[:, :-1] = map[:, 1:]

            peaks_binary = np.logical_and.reduce((map >= map_left,
                                                  map >= map_right,
                                                  map >= map_up,
                                                  map >= map_down,
                                                  map > 0.1))
            #    peaks_binary = torch.eq(
            #    peaks = zip(torch.nonzero(peaks_binary)[0],torch.nonzero(peaks_binary)[0])

            peaks = zip(np.nonzero(peaks_binary)[1],
                        np.nonzero(peaks_binary)[0])
            # note reverse
            peaks = list(peaks)

            peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
            id = range(peak_counter, peak_counter + len(peaks))
            peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in
                                       range(len(id))]

            all_peaks.append(peaks_with_score_and_id)
            peak_counter += len(peaks)

        connection_all = []
        special_k = []
        mid_num = 10

        for k in range(len(mapIdx)):
            score_mid = paf_avg[:, :, [x - 19 for x in mapIdx[k]]]
            candA = all_peaks[limbSeq[k][0] - 1]
            candB = all_peaks[limbSeq[k][1] - 1]
            nA = len(candA)
            nB = len(candB)
            indexA, indexB = limbSeq[k]
            if nA != 0 and nB != 0:
                connection_candidate = []
                for i in range(nA):
                    for j in range(nB):
                        vec = np.subtract(candB[j][:2], candA[i][:2])
                        norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                        vec = np.divide(vec, norm)

                        startend = zip(
                            np.linspace(candA[i][0], candB[j][0], num=mid_num),
                            np.linspace(candA[i][1], candB[j][1], num=mid_num))

                        startend = list(startend)

                        vec_x = np.array([score_mid[int(round(startend[I][1])),
                                                    int(round(startend[I][0])), 0]
                                          for I in range(len(startend))])
                        vec_y = np.array([score_mid[int(round(startend[I][1])),
                                                    int(round(startend[I][0])), 1]
                                          for I in range(len(startend))])

                        score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(
                            vec_y, vec[1])
                        score_with_dist_prior = sum(score_midpts) / len(
                            score_midpts) + min(0.5 * oriImg.shape[0] / norm - 1, 0)
                        criterion1 = len(
                            np.nonzero(score_midpts > 0.05)[0]) > 0.8 * len(
                            score_midpts)
                        criterion2 = score_with_dist_prior > 0
                        if criterion1 and criterion2:
                            connection_candidate.append(
                                [i, j, score_with_dist_prior,
                                 score_with_dist_prior +
                                 candA[i][2] + candB[j][2]])

                connection_candidate = sorted(connection_candidate,
                                              key=lambda x: x[2],
                                              reverse=True)
                connection = np.zeros((0, 5))
                for c in range(len(connection_candidate)):
                    i, j, s = connection_candidate[c][0:3]

                    if i not in connection[:, 3] and j not in connection[:, 4]:
                        connection = np.vstack(
                            [connection, [candA[i][3], candB[j][3], s, i, j]])

                        if len(connection) >= min(nA, nB):
                            break

                connection_all.append(connection)
            else:
                special_k.append(k)
                connection_all.append([])

        # last number in each row is the total parts number of that person
        # the second last number in each row is the score of the overall configuration
        subset = -1 * np.ones((0, 20))
        candidate = np.array(
            [item for sublist in all_peaks for item in sublist])

        for k in range(len(mapIdx)):
            if k not in special_k:
                partAs = connection_all[k][:, 0]
                partBs = connection_all[k][:, 1]
                indexA, indexB = np.array(limbSeq[k]) - 1

                for i in range(len(connection_all[k])):  # = 1:size(temp,1)
                    found = 0
                    subset_idx = [-1, -1]
                    for j in range(len(subset)):  # 1:size(subset,1):
                        if subset[j][indexA] == partAs[i] or subset[j][
                            indexB] == \
                                partBs[i]:
                            subset_idx[found] = j
                            found += 1

                    if found == 1:
                        j = subset_idx[0]

                        if subset[j][indexB] != partBs[i]:
                            subset[j][indexB] = partBs[i]
                            subset[j][-1] += 1
                            subset[j][-2] += candidate[
                                                 partBs[i].astype(int), 2] + \
                                             connection_all[k][i][2]
                    elif found == 2:  # if found 2 and disjoint, merge them
                        j1, j2 = subset_idx
                        print("found = 2")
                        membership = ((subset[j1] >= 0).astype(int) + (
                                subset[j2] >= 0).astype(int))[:-2]
                        if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                            subset[j1][:-2] += (subset[j2][:-2] + 1)
                            subset[j1][-2:] += subset[j2][-2:]
                            subset[j1][-2] += connection_all[k][i][2]
                            subset = np.delete(subset, j2, 0)
                        else:  # as like found == 1
                            subset[j1][indexB] = partBs[i]
                            subset[j1][-1] += 1
                            subset[j1][-2] += candidate[
                                                  partBs[i].astype(int), 2] + \
                                              connection_all[k][i][2]

                    # if find no partA in the subset, create a new subset
                    elif not found and k < 17:
                        row = -1 * np.ones(20)
                        row[indexA] = partAs[i]
                        row[indexB] = partBs[i]
                        row[-1] = 2
                        row[-2] = sum(
                            candidate[
                                connection_all[k][i, :2].astype(int), 2]) + \
                                  connection_all[k][i][2]
                        subset = np.vstack([subset, row])

        # delete some rows of subset which has few parts occur
        deleteIdx = []
        for i in range(len(subset)):
            if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
                deleteIdx.append(i)
        subset = np.delete(subset, deleteIdx, axis=0)

        canvas = oriImg  # B,G,R order
        for i in range(18):
            for j in range(len(all_peaks[i])):
                cv2.circle(canvas, all_peaks[i][j][0:2], 4, colors[i],
                           thickness=-1)

        stickwidth = 4

        for i in range(17):
            for n in range(len(subset)):
                index = subset[n][np.array(limbSeq[i]) - 1]
                if -1 in index:
                    continue
                cur_canvas = canvas.copy()
                Y = candidate[index.astype(int), 0]
                X = candidate[index.astype(int), 1]
                mX = np.mean(X)
                mY = np.mean(Y)
                length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
                angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
                polygon = cv2.ellipse2Poly((int(mY), int(mX)),
                                           (int(length / 2), stickwidth),
                                           int(angle), 0,
                                           360, 1)
                cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
                canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

        return canvas

    def load_model(self, ):
        models = contruct()
        model = pose_model(models)
        weight = os.path.join(os.path.abspath(os.curdir),
                              "utils/pose/pose_model.pth")
        model.load_state_dict(torch.load(weight))
        model.to(self.args.device)
        model.float()
        model.eval()
        
        return model


"""
if __name__ == '__main__':

    DEVICE = torch.device('cuda') \
        if torch.cuda.is_available() else torch.device('cpu')

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--device', default='', type=str,
                           help='device to run')

    g_args = argparser.parse_args()
    if g_args.device == '':
        g_args.device = DEVICE
    else:
        g_args.device = torch.device(DEVICE)

    im = cv2.imread('./123.jpg')
    pose = PoseEstimate(g_args)
    image = pose.process(im)
    cv2.imwrite('result.png', image)
"""
