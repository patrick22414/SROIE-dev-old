from PIL import Image
import csv
import random
import torch
from torchvision import transforms
from progress.bar import Bar

from lib_model import MyModel
from lib_data import get_train_data

RESOLUTION = 320

N_GRID = 20
GRID_SIZE = RESOLUTION // N_GRID
ANCHOR = torch.tensor(
    [[GRID_SIZE, GRID_SIZE], [GRID_SIZE, 2 * GRID_SIZE], [GRID_SIZE, 4 * GRID_SIZE]]
)
N_ANCHOR = len(ANCHOR)

model = MyModel(RESOLUTION, ANCHOR)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


def train(n_epoch, n_batch):
    at_obj = list(filter(lambda x: x % 5 == 0, range(5 * N_ANCHOR)))
    at_loc = list(filter(lambda x: x % 5 != 0, range(5 * N_ANCHOR)))

    # some fancy stuff
    bar = Bar("Train", max=n_epoch, suffix="%(percent).1f%% - %(eta)ds")

    for epoch in range(n_epoch):
        inpt, truth = get_train_data(RESOLUTION, n_batch, N_ANCHOR, N_GRID)
        oupt = model.forward(inpt)
        # objectness loss
        obj_loss = criterion(inpt[:, at_obj, :, :], truth[:, at_obj, :, :])
        # location loss
        loc_loss = criterion(inpt[:, at_loc, :, :], truth[:, at_loc, :, :])

        loss = emphasis * obj_loss + loc_loss
        loss.backward()

        optimizer.step()

        bar.next()

    bar.finish()


if __name__ == "__main__":
    train(100, 10)
