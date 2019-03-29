import csv
import os
import random

import torch
from PIL import Image, ImageColor, ImageDraw
from torchvision import transforms

from lib_data import get_train_data, get_valid_data
from lib_model import MyModel
from lib_struct import Anchor, BBox

RESOLUTION = 320

GRID_RES = 16
N_GRID = int(RESOLUTION / GRID_RES)
ANCHORS = [
    Anchor(GRID_RES * 1, GRID_RES),
    Anchor(GRID_RES * 2, GRID_RES),
    Anchor(GRID_RES * 4, GRID_RES),
    Anchor(GRID_RES * 8, GRID_RES),
    Anchor(GRID_RES * 16, GRID_RES),
]
N_ANCHOR = len(ANCHORS)

model = MyModel(RESOLUTION, ANCHORS)
crit_confidence = torch.nn.BCELoss()
crit_coordinate = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

if torch.cuda.is_available():
    model.cuda()


def train(max_epoch, batch_size):
    emphasis = RESOLUTION

    for epoch in range(max_epoch):
        optimizer.zero_grad()

        inpt, tc, tx, ty, tw, th = get_train_data(RESOLUTION, batch_size, ANCHORS, N_GRID)
        if torch.cuda.is_available():
            inpt.cuda()
            tc.cuda()
            tx.cuda()
            ty.cuda()
            tw.cuda()
            th.cuda()

        c, x, y, w, h = model.forward(inpt)

        c_1 = c[tc]
        c_0 = c[tc - 1]

        loss_c = crit_confidence(c_1, torch.ones_like(c_1)) + crit_confidence(c_0, torch.zeros_like(c_0))
        loss_x = crit_coordinate(x[tc], tx[tc])
        loss_y = crit_coordinate(y[tc], ty[tc])
        loss_w = crit_coordinate(w[tc], tw[tc])
        loss_h = crit_coordinate(h[tc], th[tc])

        loss = loss_c + loss_x + loss_y + loss_w + loss_h

        loss.backward()

        optimizer.step()

        print(">_: {:06d}: loss: {:<6.2f} ".format(epoch, loss) +
              "@c: {:<6.2f} @x: {:<6.2f} @y: {:<6.2f}".format(loss_c, loss_x, loss_y) +
              "@w: {:<6.2f} @h: {:<6.2f}".format(loss_w, loss_h))

    print(c[0, 0, :, :])
    print(tc[0, 0, :, :])


def test_1():
    train(1, 16)


if __name__ == "__main__":
    test_1()
