import numpy
import csv
from PIL import Image

H_RESO = 512  # height resolution
G_RESO = 16  # grid resolution


def txt_to_truth(txt, ratio):
    n_line = H_RESO // G_RESO
    truth = numpy.zeros([n_line, 3])
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
    offset = offset / (G_RESO / 2)
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
