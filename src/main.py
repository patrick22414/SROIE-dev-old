import argparse
import os

import numpy
import torch
from PIL import ImageDraw

from lib_data import get_train_data, get_valid_data, kmeans_anchors
from lib_model import GRID_RESO, MainModel


def train(model, args, anchors, n_grid):
    model.train()

    crit_conf = torch.nn.BCEWithLogitsLoss(reduction='mean')
    crit_coor = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(1, args.max_epoch + 1):
        optimizer.zero_grad()

        inpt, tc, tx, ty, tw, th = get_train_data(args.reso, args.batch_size, anchors, args.device)

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

        print(
            "#{:06d}: loss: {:<6.2f} @c: {:<6.2f} @x: {:<6.2f} @y: {:<6.2f} @w: {:<6.2f} @h: {:<6.2f}".format(
                epoch, loss, loss_c, loss_x, loss_y, loss_w, loss_h
            )
        )

        if args.valid_per != 0 and epoch % args.valid_per == 0:
            dirname = "../results/valid_{}/".format(epoch)
            os.mkdir(dirname)
            model.eval()
            valid_draw(model, args, dirname)
            model.train()
            print("NOTE: Valid results available at {}".format(dirname))


def valid_draw(model, args, dirname):
    with torch.no_grad():
        for i in range(10):
            # for each image
            inpt, image = get_valid_data(args.reso, args.device)
            image = image.convert("RGB")
            draw = ImageDraw.Draw(image)
            cs, xs, ys, ws, hs = model.forward(inpt)
            for ii, (c, x, y, w, h) in enumerate(zip(cs, xs, ys, ws, hs)):
                # for each tile
                mask = c > args.threshold
                c = c[mask]
                if image.width < image.height:
                    x = x[mask]
                    y = y[mask] + args.reso * ii
                else:
                    x = x[mask] + args.reso * ii
                    y = y[mask]
                w_half = w[mask] / 2
                h_half = h[mask] / 2
                boxes = numpy.stack(
                    [
                        numpy.array(x - w_half),
                        numpy.array(y - h_half),
                        numpy.array(x + w_half),
                        numpy.array(y + h_half),
                    ],
                    axis=1,
                )
                for ci, box in zip(c, boxes):
                    red = int(ci * 256)
                    draw.rectangle(tuple(box), outline=(red, 0, 0))
            image.save(dirname + "{}.jpg".format(i))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "valid", "test"], default="train")
    parser.add_argument("-d", "--device", default="cpu")
    parser.add_argument("-r", "--reso", type=int, default=512)
    parser.add_argument("-b", "--batch-size", type=int, default=16)
    parser.add_argument("-e", "--max-epoch", type=int, default=1)
    parser.add_argument("-v", "--valid-per", type=int, default=0)
    parser.add_argument("-a", "--n-anchor", type=int, default=6)
    parser.add_argument("-t", "--threshold", type=float, default=0.75)

    args = parser.parse_args()
    args.device = torch.device(args.device)

    anchors = kmeans_anchors(args.reso, args.n_anchor)

    model = MainModel(args.reso, anchors, args.device)

    n_grid = int(args.reso / GRID_RESO)

    if args.mode == "train":
        train(model, args, anchors, n_grid)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
