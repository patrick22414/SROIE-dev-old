import numpy
import csv
from PIL import Image
import torch
import os
import random
import glob


from lib_model import H_RESO, G_RESO


def get_train_data(batch_size, device):
    filenames = [os.path.splitext(f)[0] for f in glob.glob("../data_train/*.jpg")]
    samples = random.sample(filenames, batch_size)
    jpg_files = [s + ".jpg" for s in samples]
    txt_files = [s + ".txt" for s in samples]

    # convert jpg files to NCWH tensor
    data = numpy.zeros([batch_size, 3, H_RESO, H_RESO // 2], dtype=numpy.float32)
    ratio = numpy.zeros(batch_size)
    for i, f in enumerate(jpg_files):
        im = Image.open(f).convert("RGB")
        ratio[i] = H_RESO / im.height
        im = im.resize([H_RESO // 2, H_RESO])
        data[i] = numpy.moveaxis(numpy.array(im), 2, 0)

    truth = numpy.zeros([batch_size, H_RESO // G_RESO, 3], dtype=numpy.float32)
    for i, (f, r) in enumerate(zip(txt_files, ratio)):
        truth[i] = txt_to_truth(f, r)

    return torch.tensor(data, device=device), torch.tensor(truth, device=device)


def get_eval_data(batch_size, device):
    jpg_files = random.sample(glob.glob("../data_train/*.jpg"), batch_size)

    images = [Image.open(f).convert("RGB") for f in jpg_files]

    # convert jpg files to NCWH tensor
    data = numpy.zeros([batch_size, 3, H_RESO, H_RESO // 2], dtype=numpy.float32)
    for i, im in enumerate(images):
        data[i] = numpy.moveaxis(numpy.array(im.resize([H_RESO // 2, H_RESO])), 2, 0)

    return torch.tensor(data, device=device), images


def txt_to_truth(txt, ratio):
    truth = numpy.zeros([H_RESO // G_RESO, 3], dtype=numpy.float32)
    with open(txt, "r", encoding="utf-8", newline="") as file:
        for box in csv.reader(file):
            y0 = int(box[1]) * ratio
            y1 = int(box[5]) * ratio
            row = int((y0 + y1) / 2 / G_RESO)
            if truth[row, 0] == 0:
                truth[row, :] = [1, y0, y1]
            else:
                truth[row, 1] = numpy.minimum(truth[row, 1], y0)
                truth[row, 2] = numpy.maximum(truth[row, 2], y1)

    # convert y0, y1 to offset, scale
    centers = numpy.arange(G_RESO / 2, H_RESO, G_RESO)
    offset = (truth[:, 1] + truth[:, 2]) / 2 - centers
    offset = offset / G_RESO
    offset[truth[:, 0] == 0] = 0
    scale = (truth[:, 2] - truth[:, 1]) / G_RESO

    truth[:, 1] = offset
    truth[:, 2] = scale

    return truth


if __name__ == "__main__":
    numpy.set_printoptions(precision=2, suppress=True)

    txt = "../data_train/001.txt"
    jpg = "../data_train/001.jpg"
    ratio = H_RESO / Image.open(jpg).height

    truth = txt_to_truth(txt, ratio)

    print(truth)
