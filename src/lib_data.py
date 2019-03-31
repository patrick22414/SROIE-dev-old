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


def get_train_data(reso, batch_size, anchors, device):
    filenames = [os.path.splitext(f)[0] for f in glob.glob("../data_train/*.jpg")]
    samples = random.sample(filenames, batch_size)
    jpg_files = [s + ".jpg" for s in samples]
    txt_files = [s + ".txt" for s in samples]

    # convert jpg files to NCWH tensor
    data = [None] * batch_size
    ratio = [None] * batch_size
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
        data[i] = transforms.ToTensor()(image)

    # convert txt files to List of (c, x, y, w, h) of len N
    n_anchor = len(anchors)
    n_grid = int(reso / GRID_RESO)
    tc = [torch.zeros(n_anchor, n_grid, n_grid, dtype=torch.uint8) for _ in range(batch_size)]
    tx = [torch.zeros(n_anchor, n_grid, n_grid) for _ in range(batch_size)]
    ty = [torch.zeros(n_anchor, n_grid, n_grid) for _ in range(batch_size)]
    tw = [torch.zeros(n_anchor, n_grid, n_grid) for _ in range(batch_size)]
    th = [torch.zeros(n_anchor, n_grid, n_grid) for _ in range(batch_size)]
    for i, (f, r) in enumerate(zip(txt_files, ratio)):
        with open(f, "r", encoding="utf-8", newline="") as csv_file:
            for line in csv.reader(csv_file):
                l = [float(n) * r for n in line[0:8]]
                box_x = (l[0] + l[4]) / 2 - crop_at[0]
                box_y = (l[1] + l[5]) / 2 - crop_at[1]
                if box_x < 0 or box_x >= reso or box_y < 0 or box_y >= reso:
                    continue
                box_w = l[4] - l[0]
                box_h = l[5] - l[1]
                grid_x = int(box_x / GRID_RESO)
                grid_y = int(box_y / GRID_RESO)
                anchor_choice = best_anchor(box_w, box_h, anchors)
                if tc[i][anchor_choice, grid_y, grid_x].item() == 0:
                    tc[i][anchor_choice, grid_y, grid_x] = 1
                    tx[i][anchor_choice, grid_y, grid_x] = box_x
                    ty[i][anchor_choice, grid_y, grid_x] = box_y
                    tw[i][anchor_choice, grid_y, grid_x] = box_w
                    th[i][anchor_choice, grid_y, grid_x] = box_h

    return (
        torch.stack(data, dim=0).to(device),
        torch.stack(tc, dim=0).to(device),
        torch.stack(tx, dim=0).to(device),
        torch.stack(ty, dim=0).to(device),
        torch.stack(tw, dim=0).to(device),
        torch.stack(th, dim=0).to(device),
    )


def get_valid_data(reso, device):
    jpg_file = random.choice(glob.glob("../data_train/*.jpg"))
    image = Image.open(jpg_file).convert("L")

    if image.width < image.height:
        ratio = reso / image.width
        new_size = (reso, int(image.height * ratio))
        n_tile = int(numpy.ceil(new_size[1] / reso))
        image = image.resize(new_size)
        image = transforms.functional.pad(image, (0, 0, 0, n_tile * reso - image.height), fill=255)

        tensor = torch.zeros(n_tile, 1, reso, reso, device=device)
        for i in range(n_tile):
            tensor[i] = transforms.ToTensor()(image.crop((0, i * reso, reso, (i + 1) * reso)))
    else:
        ratio = reso / image.height
        new_size = (int(image.width * ratio), reso)
        n_tile = int(numpy.ceil(new_size[0] / reso))
        image = image.resize(new_size)
        image = transforms.functional.pad(image, (0, 0, n_tile * reso - image.width, 0), fill=255)

        tensor = torch.zeros(n_tile, 1, reso, reso, device=device)
        for i in range(n_tile):
            tensor[i] = transforms.ToTensor()(image.crop((i * reso, 0, (i + 1) * reso, reso)))

    return tensor, image


def best_anchor(w, h, anchors):
    dist_w = numpy.array([a[0] for a in anchors]) - w
    dist_h = numpy.array([a[1] for a in anchors]) - h
    return numpy.argmin(numpy.hypot(dist_w, dist_h))


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
    tensor, image = get_valid_data(480, torch.device("cpu"))
    print(tensor)
    image.save("../tmp/tmp.jpg")
