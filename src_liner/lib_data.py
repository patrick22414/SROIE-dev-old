import numpy
import csv
from PIL import Image
import torch
import os
import random
import glob


from lib_model import RESO_H, RESO_W, GRID_H, GRID_W


def get_train_data(batch_size, device):
    filenames = [os.path.splitext(f)[0] for f in glob.glob("../data_train/*.jpg")]
    samples = random.sample(filenames, batch_size)
    jpg_files = [s + ".jpg" for s in samples]
    txt_files = [s + ".txt" for s in samples]

    # convert jpg files to NCWH tensor
    data = numpy.zeros([batch_size, 3, RESO_H, RESO_W], dtype=numpy.float32)
    ratio = numpy.zeros(batch_size)
    for i, f in enumerate(jpg_files):
        im = Image.open(f).convert("RGB")
        ratio[i] = RESO_H / im.height
        im = im.resize([RESO_W, RESO_H])
        data[i] = numpy.moveaxis(numpy.array(im), 2, 0)

    truth = numpy.zeros([batch_size, RESO_H // GRID_H, 3], dtype=numpy.float32)
    for i, (f, r) in enumerate(zip(txt_files, ratio)):
        truth[i] = txt_to_truth(f, r)

    return torch.tensor(data, device=device), torch.tensor(truth, device=device)


def get_train_data2(batch_size, device):
    filenames = [os.path.splitext(f)[0] for f in glob.glob("../data_train/*.jpg")]
    samples = random.sample(filenames, batch_size)
    jpg_files = [s + ".jpg" for s in samples]
    txt_files = [s + ".txt" for s in samples]

    # convert jpg files to NCWH tensor
    data = numpy.zeros([batch_size, 3, RESO_H, RESO_W], dtype=numpy.float32)
    ratio = numpy.zeros([batch_size, 2])
    for i, f in enumerate(jpg_files):
        im = Image.open(f).convert("RGB")
        ratio[i] = (RESO_W / im.width, RESO_H / im.height)
        im = im.resize([RESO_W, RESO_H])
        data[i] = numpy.moveaxis(numpy.array(im), 2, 0)

    truth = numpy.zeros([batch_size, 5, RESO_H // GRID_H, RESO_W // GRID_W], dtype=numpy.float32)
    for i, (f, r) in enumerate(zip(txt_files, ratio)):
        truth[i] = txt_to_truth2(f, r)

    return torch.tensor(data, device=device), torch.tensor(truth, device=device)


def get_eval_data(batch_size, device):
    jpg_files = random.sample(glob.glob("../data_valid/*.jpg"), batch_size)

    images = [Image.open(f).convert("RGB") for f in jpg_files]

    # convert jpg files to NCWH tensor
    data = numpy.zeros([batch_size, 3, RESO_H, RESO_W], dtype=numpy.float32)
    for i, im in enumerate(images):
        data[i] = numpy.moveaxis(numpy.array(im.resize([RESO_W, RESO_H])), 2, 0)

    return torch.tensor(data, device=device), images


def get_eval_data2(batch_size, device):
    jpg_files = random.sample(glob.glob("../data_valid/*.jpg"), batch_size)

    images = [Image.open(f).convert("RGB") for f in jpg_files]

    # convert jpg files to NCWH tensor
    data = numpy.zeros([batch_size, 3, RESO_H, RESO_W], dtype=numpy.float32)
    for i, im in enumerate(images):
        data[i] = numpy.moveaxis(numpy.array(im.resize([RESO_W, RESO_H])), 2, 0)

    return torch.tensor(data, device=device), images


def txt_to_truth(txt, ratio):
    truth = numpy.zeros([RESO_H // GRID_H, 3], dtype=numpy.float32)
    with open(txt, "r", encoding="utf-8", newline="") as file:
        for box in csv.reader(file):
            y0 = int(box[1]) * ratio
            y1 = int(box[5]) * ratio
            row = int((y0 + y1) / 2 / GRID_H)
            if truth[row, 0] == 0:
                truth[row, :] = [1, y0, y1]
            else:
                truth[row, 1] = numpy.minimum(truth[row, 1], y0)
                truth[row, 2] = numpy.maximum(truth[row, 2], y1)

    # convert y0, y1 to offset, scale
    centers = numpy.arange(GRID_H / 2, RESO_H, GRID_H)
    offset = (truth[:, 1] + truth[:, 2]) / 2 - centers
    offset = offset / GRID_H
    offset[truth[:, 0] == 0] = 0
    scale = (truth[:, 2] - truth[:, 1]) / GRID_H

    truth[:, 1] = offset
    truth[:, 2] = scale

    return truth


def txt_to_truth2(txt, ratio):
    truth = numpy.zeros([5, RESO_H // GRID_H, RESO_W//GRID_W], dtype=numpy.float32)
    with open(txt, "r", encoding="utf-8", newline="") as file:
        for box in csv.reader(file):
            x0 = int(box[0]) * ratio[0]
            y0 = int(box[1]) * ratio[1]
            x1 = int(box[4]) * ratio[0]
            y1 = int(box[5]) * ratio[1]
            col = int((x0 + x1) / 2 / GRID_W)
            row = int((y0 + y1) / 2 / GRID_H)
            if truth[0, row, col] == 0:
                truth[:, row, col] = [1, x0, y0, x1, y1]
            else:
                truth[1, row, col] = numpy.minimum(truth[1, row, col], x0)
                truth[2, row, col] = numpy.minimum(truth[2, row, col], y0)
                truth[3, row, col] = numpy.maximum(truth[3, row, col], x1)
                truth[4, row, col] = numpy.maximum(truth[4, row, col], y1)

    center_x = numpy.stack([numpy.arange(GRID_W // 2, RESO_W, GRID_W)] * (RESO_H // GRID_H), axis=0)
    center_y = numpy.stack([numpy.arange(GRID_H // 2, RESO_H, GRID_H)] * (RESO_W // GRID_W), axis=1)
    offset_x = ((truth[1] + truth[3]) / 2 - center_x) / GRID_W
    offset_y = ((truth[2] + truth[4]) / 2 - center_y) / GRID_H
    scale_x = (truth[3] - truth[1]) / GRID_W
    scale_y = (truth[4] - truth[2]) / GRID_H

    mask = truth[0] == 0
    offset_x[mask] = 0
    offset_y[mask] = 0
    scale_x[mask] = 0
    scale_y[mask] = 0

    truth[1] = offset_x
    truth[2] = offset_y
    truth[3] = scale_x
    truth[4] = scale_y

    return truth


if __name__ == "__main__":
    numpy.set_printoptions(precision=2, edgeitems=10, suppress=True)

    txt = "../data_train/001.txt"
    jpg = "../data_train/001.jpg"
    image = Image.open(jpg)
    ratio = (RESO_W / image.width, RESO_H / image.height)

    truth = txt_to_truth2(txt, ratio)

    print(truth)
