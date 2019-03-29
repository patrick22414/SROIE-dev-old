import csv
import glob
import os
import random

import numpy
import torch
from PIL import Image
from torchvision import transforms


def get_train_data(res, batch_size, anchors, n_grid):
    samples = random.sample(range(600), batch_size)
    jpg_files = ["../data_train/{:03d}.jpg".format(s) for s in samples]
    txt_files = ["../data_train/{:03d}.txt".format(s) for s in samples]

    # convert jpg files to NCWH tensor
    images = [Image.open(file).convert("L") for file in jpg_files]
    ratio_x = [res / im.width for im in images]
    ratio_y = [res / im.height for im in images]
    transform = transforms.Compose([transforms.Resize((res, res)), transforms.ToTensor()])
    data = torch.stack(list(map(transform, images)), dim=0)

    # convert txt files to List of (c, x, y, w, h) of len N
    tc = [None] * batch_size
    tx = [None] * batch_size
    ty = [None] * batch_size
    tw = [None] * batch_size
    th = [None] * batch_size
    for i, (f, rx, ry) in enumerate(zip(txt_files, ratio_x, ratio_y)):
        tc[i], tx[i], ty[i], tw[i], th[i], = txt_to_tensors(f, rx, ry, anchors, n_grid, int(res / n_grid))

    return (
        data,
        torch.stack(tc, dim=0),
        torch.stack(tx, dim=0),
        torch.stack(ty, dim=0),
        torch.stack(tw, dim=0),
        torch.stack(th, dim=0),
    )


def get_valid_data(res, batch_size):
    jpg_files = random.sample(glob.glob("../data_valid/*.jpg"), batch_size)

    # convert jpg files to NCWH
    images = [Image.open(file).convert("L") for file in jpg_files]
    transform = transforms.Compose([transforms.Resize((res, res), Image.BICUBIC), transforms.ToTensor()])
    tensor = torch.stack(list(map(transform, images)), dim=0)

    return tensor, jpg_files


def best_anchor(w, h, anchors):
    dist_w = numpy.array([a.w for a in anchors]) - w
    dist_h = numpy.array([a.h for a in anchors]) - h
    return numpy.argmin(numpy.hypot(dist_w, dist_h))


def txt_to_tensors(txt_file, ratio_x, ratio_y, anchors, n_grid, grid_res):
    n_anchor = len(anchors)

    c = torch.zeros(n_anchor, n_grid, n_grid, dtype=torch.uint8)
    x = torch.zeros(n_anchor, n_grid, n_grid)
    y = torch.zeros(n_anchor, n_grid, n_grid)
    w = torch.zeros(n_anchor, n_grid, n_grid)
    h = torch.zeros(n_anchor, n_grid, n_grid)

    with open(txt_file, "r", encoding="utf-8", newline="") as csv_file:
        for line in csv.reader(csv_file):
            l = [int(n) for n in line[0:8]]
            box_x = (l[0] + l[4]) / 2 * ratio_x
            box_y = (l[1] + l[5]) / 2 * ratio_y
            box_w = (l[4] - l[0]) * ratio_x
            box_h = (l[5] - l[1]) * ratio_y
            grid_x = int(box_x / grid_res)
            grid_y = int(box_y / grid_res)
            anchor_choice = best_anchor(box_w, box_h, anchors)
            if c[anchor_choice, grid_y, grid_x].item() == 0:
                c[anchor_choice, grid_y, grid_x] = 1
                x[anchor_choice, grid_y, grid_x] = box_x
                y[anchor_choice, grid_y, grid_x] = box_y
                w[anchor_choice, grid_y, grid_x] = box_w
                h[anchor_choice, grid_y, grid_x] = box_h

    return c, x, y, w, h


if __name__ == "__main__":
    pass
