import csv
import os
import random

import torch
from PIL import Image, ImageColor, ImageDraw
from progress.bar import Bar
from torchvision import transforms

from lib_data import get_train_data, get_valid_data
from lib_model import MyModel
from lib_struct import Anchor, BBox

RESOLUTION = 320

N_GRID = 20
GRID_RES = RESOLUTION // N_GRID
ANCHOR = [Anchor(GRID_RES, GRID_RES), Anchor(GRID_RES * 2, GRID_RES), Anchor(GRID_RES * 4, GRID_RES)]
N_ANCHOR = len(ANCHOR)

model = MyModel(RESOLUTION, ANCHOR)
criterion_obj = torch.nn.BCELoss()
criterion_loc = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


def train(n_epoch, n_batch):
    at_obj = list(filter(lambda x: x % 5 == 0, range(5 * N_ANCHOR)))
    at_loc = list(filter(lambda x: x % 5 != 0, range(5 * N_ANCHOR)))

    emphasis = RESOLUTION

    for epoch in range(n_epoch):
        inpt, truth = get_train_data(RESOLUTION, n_batch, N_ANCHOR, N_GRID)
        oupt = model.forward(inpt)
        # print(oupt)

        # objectness loss
        obj_loss = criterion_obj(oupt[:, at_obj, :, :], truth[:, at_obj, :, :])
        # location loss
        loc_loss = criterion_loc(oupt[:, at_loc, :, :], truth[:, at_loc, :, :])

        loss = obj_loss * emphasis + loc_loss / emphasis
        print(f"{epoch:06d}: {obj_loss.item():.2f}, \t{loc_loss.item():.2f}, \t{loss.item():.2f}")

        loss.backward()

        optimizer.step()


def valid_draw(dirname, n_batch, threshold):
    with torch.no_grad():
        inpt, jpg_files = get_valid_data(RESOLUTION, n_batch)
        oupt = (
            model.forward(inpt)
            .reshape(n_batch, 5 * N_ANCHOR, N_GRID * N_GRID)
            .transpose(1, 2)
            .reshape(n_batch, N_GRID * N_GRID * N_ANCHOR, 5)
        )

        # print(oupt)

    for file, res in zip(jpg_files, oupt):
        canvas = Image.open(file).convert("L")
        up_x = canvas.width / RESOLUTION
        up_y = canvas.height / RESOLUTION
        draw = ImageDraw.Draw(canvas)
        for box in res:
            if box[0].item() > threshold:
                print(box[0].item())
                draw.rectangle(box_tensor_to_rect(box, up_x, up_y), outline=0)
            # else:
            #     print(box[0].item())
        canvas.save(dirname + os.path.basename(file))


def box_tensor_to_rect(box, up_x, up_y):
    x = box[1] * up_x
    y = box[2] * up_y
    wh = box[3] / 2 * up_x
    hh = box[4] / 2 * up_y
    return [x - wh, y - hh, x + wh, y + hh]


def test_100():
    max_epoch = 100
    train(max_epoch, 16)
    valid_draw(f"../results/valid_100/", 10, 0.6)


def test_1():
    train(1, 10)


if __name__ == "__main__":
    test_100()
