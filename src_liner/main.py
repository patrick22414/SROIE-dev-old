import glob
import os
import torch
import argparse

from lib_model import Model
from lib_data import get_train_data
from lib_draw import draw_prediction

H_RESO = 512  # height resolution
G_RESO = 16  # grid resolution


def train(model, args):
    model.to(args.device)
    model.train()

    bce_loss = torch.nn.BCEWithLogitsLoss()
    mse_loss = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1000)

    for epoch in range(1, args.max_epoch + 1):
        optimizer.zero_grad()

        data, truth = get_train_data()
        data = data.to(args.device)
        truth = truth.to(args.device)

        preds = model(data)

        loss_c = bce_loss(preds[:, :, 0], truth[:, :, 0])
        loss_o = mse_loss(preds[:, :, 1], truth[:, :, 2])
        loss_s = mse_loss(preds[:, :, 2], truth[:, :, 2])

        loss = loss_c + loss_o + loss_s
        loss.backward()

        optimizer.step()
        scheduler.step()

        if args.eval_per != 0 and epoch % args.eval_per == 0:
            eval_data, _ = get_train_data()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", default="cpu")
    parser.add_argument("-b", "--batch-size", type=int, default=16)
    parser.add_argument("-e", "--max-epoch", type=int, default=1)
    parser.add_argument("-v", "--eval-per", type=int, default=0)

    args = parser.parse_args()
    args.device = torch.device(args.device)

    model = Model()
    train(model, args)


if __name__ == "__main__":
    main()
