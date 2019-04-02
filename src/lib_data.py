import csv
import glob
import os
import random

import numpy
import torch
from PIL import Image
from scipy.cluster.vq import kmeans
from torchvision import transforms

from lib_model import GRID_RESO


def get_train_data(reso, batch_size, anchors, ignore, device):
    filenames = [os.path.splitext(f)[0] for f in glob.glob("../data_train/*.jpg")]
    samples = random.sample(filenames, batch_size)
    jpg_files = [s + ".jpg" for s in samples]
    txt_files = [s + ".txt" for s in samples]

    # convert jpg files to NCWH tensor
    data = numpy.zeros([batch_size, 3, reso, reso], dtype=numpy.float32)
    ratio = numpy.zeros(batch_size)
    for i, jpg in enumerate(jpg_files):
        image = Image.open(jpg).convert("RGB")

        if image.width < image.height:
            ratio[i] = reso / image.width
            new_size = (reso, int(image.height * ratio[i]))
            crop_at = (0, random.randint(0, new_size[1] - reso))
        else:
            ratio[i] = reso / image.height
            new_size = (int(image.width * ratio[i]), reso)
            crop_at = (random.randint(0, new_size[0] - reso), 0)

        crop_at = crop_at + (crop_at[0] + reso, crop_at[1] + reso)
        image = image.resize(new_size).crop(crop_at)
        data[i] = numpy.rollaxis(numpy.array(image), 2, 0)

    # convert txt files to List of (c, x, y, w, h) of len N
    n_anchor = len(anchors)
    n_grid = int(reso / GRID_RESO)
    maskc = [torch.ones(n_anchor, n_grid, n_grid) for _ in range(batch_size)]
    tc = numpy.zeros([batch_size, n_anchor, n_grid, n_grid], dtype=numpy.uint8)
    tx = numpy.zeros([batch_size, n_anchor, n_grid, n_grid], dtype=numpy.float32)
    ty = numpy.zeros([batch_size, n_anchor, n_grid, n_grid], dtype=numpy.float32)
    tw = numpy.zeros([batch_size, n_anchor, n_grid, n_grid], dtype=numpy.float32)
    th = numpy.zeros([batch_size, n_anchor, n_grid, n_grid], dtype=numpy.float32)
    maskc = numpy.ones([batch_size, n_anchor, n_grid, n_grid], dtype=numpy.uint8)
    for i, (f, r) in enumerate(zip(txt_files, ratio)):
        # print("Converting txt", f)
        with open(f, "r", encoding="utf-8", newline="") as csv_file:
            for line in csv.reader(csv_file):
                l = [int(n) * r for n in line[0:8]]
                x = (l[0] + l[4]) / 2 - crop_at[0]
                y = (l[1] + l[5]) / 2 - crop_at[1]
                if x < 0 or x >= reso or y < 0 or y >= reso:
                    continue
                w = l[4] - l[0]
                h = l[5] - l[1]
                box = numpy.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2])
                ious, best, g0x, g0y, g1x, g1y = best_anchors(reso, anchors, box)
                best = (i,) + best
                # grid_x = int(x / GRID_RESO)
                # grid_y = int(y / GRID_RESO)
                # anchor_choice = best_anchor(w, h, anchors)
                # print(best, tc.shape, g0y, g0x)
                if tc[best] == 0:
                    tc[best] = 1
                    tx[best] = x
                    ty[best] = y
                    tw[best] = w
                    th[best] = h
                    # print("ious:", ious)
                    # print("shape 1:", maskc[i, :, g0y:g1y, g0x:g1x].shape)
                    # print("shape 2:", ious.shape)
                    maskc[i, :, g0y:g1y, g0x:g1x][ious > ignore] = 0

    maskc = maskc + tc

    # print(maskc[0, 0, ...])
    # print(tc[0, 0, ...])

    return (
        torch.tensor(data, device=device),
        torch.tensor(tc, device=device),
        torch.tensor(tx, device=device),
        torch.tensor(ty, device=device),
        torch.tensor(tw, device=device),
        torch.tensor(th, device=device),
        torch.tensor(maskc, device=device),
    )


def get_valid_data(reso, device):
    jpg_file = random.choice(glob.glob("../data_train/*.jpg"))
    image = Image.open(jpg_file).convert("RGB")

    if image.width < image.height:
        ratio = reso / image.width
        new_size = (reso, int(image.height * ratio))
        n_tile = int(numpy.ceil(new_size[1] / reso))
        image = image.resize(new_size)
        image = transforms.functional.pad(image, (0, 0, 0, n_tile * reso - image.height), fill=(255, 255, 255))

        tensor = torch.zeros(n_tile, 3, reso, reso, device=device)
        for i in range(n_tile):
            tensor[i] = transforms.ToTensor()(image.crop((0, i * reso, reso, (i + 1) * reso)))
    else:
        ratio = reso / image.height
        new_size = (int(image.width * ratio), reso)
        n_tile = int(numpy.ceil(new_size[0] / reso))
        image = image.resize(new_size)
        image = transforms.functional.pad(image, (0, 0, n_tile * reso - image.width, 0), fill=(255, 255, 255))

        tensor = torch.zeros(n_tile, 3, reso, reso, device=device)
        for i in range(n_tile):
            tensor[i] = transforms.ToTensor()(image.crop((i * reso, 0, (i + 1) * reso, reso)))

    return tensor, image


def best_anchors(reso, anchors, truth_box):
    # print("truth_box:", truth_box)
    n_grid = int(reso / GRID_RESO)
    g0x = int(numpy.floor(truth_box[0] / GRID_RESO))
    g0y = int(numpy.floor(truth_box[1] / GRID_RESO))
    g1x = int(numpy.ceil(truth_box[2] / GRID_RESO))
    g1y = int(numpy.ceil(truth_box[3] / GRID_RESO))

    g0x = numpy.maximum(g0x, 0)
    g0y = numpy.maximum(g0y, 0)
    g1x = numpy.minimum(g1x, n_grid)
    g1y = numpy.minimum(g1y, n_grid)

    n_anchor = len((anchors))
    anchor_boxes = numpy.zeros([n_anchor, g1y - g0y, g1x - g0x, 4])

    for k, a in enumerate(anchors):
        for j in range(g1y - g0y):
            for i in range(g1x - g0x):
                center_x = (i + g0x + 0.5) * GRID_RESO
                center_y = (j + g0y + 0.5) * GRID_RESO
                anchor_boxes[k, j, i, :] = numpy.array(
                    [center_x - a[0] / 2, center_y - a[1] / 2, center_x + a[0] / 2, center_y + a[1] / 2]
                )

    # print(anchor_boxes)
    ious = iou_with_truth(anchor_boxes.reshape(-1, 4), truth_box).reshape(n_anchor, g1y - g0y, g1x - g0x)

    # print("max iou:", numpy.max(ious))
    best = numpy.argmax(ious)
    best = numpy.unravel_index(best, ious.shape)
    best = (best[0], best[1] + g0y, best[2] + g0x)
    # print("gs:", g0x, g0y, g1x, g1y)

    return ious, best, g0x, g0y, g1x, g1y


def iou_with_truth(anchor_boxes, truth_box):
    m = numpy.maximum(anchor_boxes, truth_box)
    n = numpy.minimum(anchor_boxes, truth_box)
    return ((n[:, 2] - m[:, 0]) * (n[:, 3] - m[:, 1])) / ((m[:, 2] - n[:, 0]) * (m[:, 3] - n[:, 1]))


def kmeans_anchors(reso, n_anchor):
    filenames = [os.path.splitext(f)[0] for f in glob.glob("../data_train/*.jpg")]
    jpg_files = [f + ".jpg" for f in filenames]
    txt_files = [f + ".txt" for f in filenames]

    observation = [None] * len(jpg_files)
    for i, (jpg, txt) in enumerate(zip(jpg_files, txt_files)):
        # print(jpg)
        image = Image.open(jpg)
        ratio = reso / image.width

        with open(txt, "r", encoding="utf-8", newline="") as csv_file:
            lines = numpy.array([[int(x) for x in line[0:8]] for line in csv.reader(csv_file)])
            wh = (lines[:, [4, 5]] - lines[:, [0, 1]]) * ratio
            observation[i] = wh

    observation = numpy.concatenate(observation, axis=0).astype(numpy.float64)
    anchors, distortion = kmeans(observation, n_anchor)

    print("NOTE: Find {} anchors\n{} with distortion {}".format(n_anchor, anchors, distortion))
    return anchors


if __name__ == "__main__":
    numpy.set_printoptions(threshold=10000, edgeitems=10, linewidth=1000)
    reso = 512
    anchors = kmeans_anchors(reso, 6)
    get_train_data(reso, 1, anchors, 0.5, torch.device("cpu"))
