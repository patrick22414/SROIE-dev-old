import argparse
import torch

from lib_struct import Anchor
from lib_model import MyModel
from lib_data import get_train_data, get_valid_data

GRID_RES = 16


def train(model, args, anchors, n_grid):
    model.train()

    crit_conf = torch.nn.BCELoss()
    crit_coor = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    for epoch in range(args.max_epoch):
        optimizer.zero_grad()

        inpt, tc, tx, ty, tw, th = get_train_data(args.resolution,
                                                  args.batch_size,
                                                  anchors,
                                                  n_grid,
                                                  args.device)

        c, x, y, w, h = model.forward(inpt)

        c_1 = c[tc]
        c_0 = c[tc - 1]

        loss_c = crit_conf(c_1, torch.ones_like(c_1)) + crit_conf(c_0, torch.zeros_like(c_0))
        loss_x = crit_coor(x[tc], tx[tc])
        loss_y = crit_coor(y[tc], ty[tc])
        loss_w = crit_coor(w[tc], tw[tc])
        loss_h = crit_coor(h[tc], th[tc])

        loss = loss_c + loss_x + loss_y + loss_w + loss_h

        loss.backward()

        optimizer.step()

        print("#{:06d}: loss: {:<6.2f} ".format(epoch, loss) +
              "@c: {:<6.2f} @x: {:<6.2f} @y: {:<6.2f}".format(loss_c, loss_x, loss_y) +
              "@w: {:<6.2f} @h: {:<6.2f}".format(loss_w, loss_h))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "valid", "test"], default="train")
    parser.add_argument("-d", "--device", default="cpu")
    parser.add_argument("-r", "--resolution", type=int, default=320)
    parser.add_argument("-b", "--batch-size", type=int, default=10)
    parser.add_argument("-e", "--max-epoch", type=int, default=1)

    args = parser.parse_args()
    args.device = torch.device(args.device)

    n_grid = int(args.resolution / GRID_RES)
    anchors = [
        Anchor(GRID_RES * 1, GRID_RES),
        Anchor(GRID_RES * 2, GRID_RES),
        Anchor(GRID_RES * 4, GRID_RES),
        Anchor(GRID_RES * 8, GRID_RES),
        Anchor(GRID_RES * 16, GRID_RES),
    ]

    model = MyModel(args.resolution, anchors, args.device)
    if args.mode == "train":
        train(model, args, anchors, n_grid)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
