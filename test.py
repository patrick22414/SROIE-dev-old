import argparse
import csv
import glob
import os
import sys
import random

import numpy
import torch
from bokeh import plotting
from PIL import Image
from scipy.cluster.vq import kmeans, whiten
from torchvision import transforms


def my_function(x):
    t = [None] * 3
    t[0] = x[0, :, :].add(10)
    t[1] = x[1, :, :].mul(0.5)
    t[2] = x[2, :, :].exp()
    return torch.stack(t, dim=0)


def kmeans_anchors():
    filenames = [os.path.splitext(f)[0] for f in glob.glob("data_train/*.jpg")]
    jpg_files = [f + ".jpg" for f in filenames]
    txt_files = [f + ".txt" for f in filenames]

    fig = plotting.figure()

    reso = 320
    observation = [None] * len(jpg_files)
    for i, (jpg, txt) in enumerate(zip(jpg_files, txt_files)):
        # print(jpg)
        image = Image.open(jpg)
        ratio = reso / image.width

        with open(txt, "r", encoding="utf-8", newline="") as csv_file:
            lines = numpy.array([[int(x) for x in line[0:8]] for line in csv.reader(csv_file)])
            wh = lines[:, [4, 5]] - lines[:, [0, 1]]
            wh = wh * ratio
            observation[i] = wh

    observation = numpy.concatenate(observation, axis=0).astype(float)
    centroids, distortion = kmeans(observation, 6, iter=100)
    fig.cross(observation[:, 0], observation[:, 1], line_color="skyblue")
    fig.circle(centroids[:, 0], centroids[:, 1], fill_color="orange")
    print(centroids, distortion)
    # plotting.show(fig)


def random_square():
    filenames = [os.path.splitext(f)[0] for f in glob.glob("data_train/*.jpg")]
    jpg_files = [f + ".jpg" for f in filenames]
    txt_files = [f + ".txt" for f in filenames]

    reso = 480
    transform = transforms.Compose([transforms.Resize(reso), transforms.RandomCrop(reso)])

    random.seed(0)

    for jpg in random.sample(jpg_files, 10):
        image = Image.open(jpg).convert("L")
        if image.width < image.height:
            new_size = (reso, int(image.height * reso / image.width))
            crop_at = (0, random.randint(0, new_size[1] - reso))
        else:
            new_size = (int(image.width * reso / image.height), reso)
            crop_at = (random.randint(0, new_size[0] - reso), 0)
        crop_at = crop_at + (crop_at[0] + reso, crop_at[1] + reso)

        image = image.resize(new_size).crop(crop_at)
        image = transforms.ToTensor()(image)
        print(image)


def move_to_data_valid():
    poor_jpgs = random.sample(glob.glob("data_train/*.jpg"), 50)
    for f in poor_jpgs:
        new_name = "data_valid/" + os.path.basename(f)
        os.rename(f, new_name)


if __name__ == "__main__":
    move_to_data_valid()
