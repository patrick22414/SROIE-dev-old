import glob
import os
from csv import reader

import numpy
from matplotlib import pyplot
from PIL import Image


def image_ratio_hist():
    jpg_files = glob.glob("../data_train/*.jpg")

    ratios = numpy.zeros(len(jpg_files))
    for i, f in enumerate(jpg_files):
        im = Image.open(f)
        ratios[i] = im.height / im.width

    print(ratios)

    pyplot.hist(ratios, bins=100)
    pyplot.title("Aspect Ratios")
    pyplot.show()


def text_box_ratio_hist():
    txt_files = glob.glob("../data_train/*.txt")

    width_height = [None] * len(txt_files)

    for i, txt in enumerate(txt_files):
        with open(txt, "r", encoding="utf-8", newline="") as csv:
            width_height[i] = numpy.zeros((sum(1 for line in reader(csv)), 2))

            csv.seek(0)
            for ii, line in enumerate(reader(csv)):
                w = int(line[4]) - int(line[0])
                h = int(line[5]) - int(line[1])
                if h > 400:
                    print(txt)
                width_height[i][ii, :] = numpy.array([w, h])

        if numpy.count_nonzero(width_height[i]) != width_height[i].size:
            print(txt)
            print(width_height[i])

    width_height = numpy.concatenate(width_height, axis=0)

    pyplot.scatter(width_height[:, 0], width_height[:, 1], alpha=0.1)
    pyplot.show()


if __name__ == "__main__":
    text_box_ratio_hist()
