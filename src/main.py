import argparse
import datetime
import glob
import os
import time

import numpy
import torch

from lib_data import get_eval_data, get_eval_data2, get_train_data, get_train_data2
from lib_draw import draw_pred_grid, draw_pred_line
from lib_model import LineModel, GridModel


def train_line(model, args):
    model.to(args.device)
    model.train()

    bce_loss = torch.nn.BCEWithLogitsLoss()
    mse_loss = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2000)

    avg_loss = 0
    loss_history = numpy.zeros(args.max_epoch)
    loss_history_file = "../loss_history/{}.txt".format(
        datetime.datetime.now().strftime("LINE-%Y%m%d-%H%M")
    )
    os.makedirs("../loss_history/", exist_ok=True)

    for epoch in range(1, args.max_epoch + 1):
        start = time.time()

        optimizer.zero_grad()

        data, truth = get_train_data(args.batch_size, args.device)
        mask = truth[:, :, 0].byte()

        preds = model(data)

        loss_c = bce_loss(preds[:, :, 0], truth[:, :, 0])
        loss_o = mse_loss(preds[:, :, 1][mask], truth[:, :, 1][mask])
        loss_s = mse_loss(preds[:, :, 2][mask], truth[:, :, 2][mask])

        loss = loss_c + loss_o + loss_s
        loss.backward()

        optimizer.step()
        scheduler.step()

        avg_loss = 0.9 * avg_loss + 0.1 * loss.item()
        loss_history[epoch - 1] = avg_loss

        print(
            "#{:04d} ".format(epoch),
            "| LR: {:.1e}".format(scheduler.get_lr()[0]),
            "| Loss: {:.2e} ({:.2e}, {:.2e}, {:.2e})".format(
                avg_loss, loss_c, loss_o, loss_s
            ),
            "| Range: ({:.2e}, {:.2e})".format(
                torch.sigmoid(preds[:, :, 0].min()).item(),
                torch.sigmoid(preds[:, :, 0].max()).item(),
            ),
            "| T: {:4.2f}s".format(time.time() - start),
        )

        if args.checkpoint != 0 and epoch % args.checkpoint == 0:
            numpy.savetxt(loss_history_file, loss_history[0:epoch])
            print("NOTE: Loss history available at {}".format(loss_history_file))

        if args.eval_per != 0 and epoch % args.eval_per == 0:
            with torch.no_grad():
                dirname = "../results_line/eval_{}/".format(epoch)
                os.makedirs(dirname, exist_ok=True)

                model.eval()

                eval_data, eval_images = get_eval_data(args.batch_size, args.device)
                eval_preds = model(eval_data)

                if eval_preds.is_cuda:
                    eval_preds = eval_preds.cpu().numpy()
                else:
                    eval_preds = eval_preds.numpy()

                for i, (pred, image) in enumerate(zip(eval_preds, eval_images)):
                    draw_pred_line(image, pred)
                    image.save(dirname + "{}.png".format(i))

                print("NOTE: Eval result available at {}".format(dirname))

                model.train()


def train2(model, args):
    model.to(args.device)
    model.train()

    bce_loss = torch.nn.BCEWithLogitsLoss()
    mse_loss = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 4000)

    avg_loss = 0
    loss_history = numpy.zeros(args.max_epoch)
    loss_history_file = "../loss_history/{}.txt".format(
        datetime.datetime.now().strftime("%Y%m%d-%H%M")
    )
    os.makedirs("../loss_history/", exist_ok=True)

    for epoch in range(1, args.max_epoch + 1):
        start = time.time()

        optimizer.zero_grad()

        data, truth = get_train_data2(args.batch_size, args.device)
        mask = truth[:, 0, :, :].byte()

        preds = model(data)

        loss_c = bce_loss(preds[:, 0, :, :], truth[:, 0, :, :])
        loss_ox = mse_loss(preds[:, 1, :, :][mask], truth[:, 1, :, :][mask])
        loss_oy = mse_loss(preds[:, 2, :, :][mask], truth[:, 2, :, :][mask])
        loss_sx = mse_loss(preds[:, 3, :, :][mask], truth[:, 3, :, :][mask])
        loss_sy = mse_loss(preds[:, 4, :, :][mask], truth[:, 4, :, :][mask])

        loss = loss_c + loss_ox + loss_oy + loss_sx + loss_sy
        loss.backward()

        optimizer.step()
        scheduler.step()

        avg_loss = 0.9 * avg_loss + 0.1 * loss.item()
        loss_history[epoch - 1] = avg_loss

        print(
            "#{:04d} ".format(epoch),
            "| LR: {:.1e}".format(scheduler.get_lr()[0]),
            "| Loss: {:.2e} ({:.2e}, {:.2e}, {:.2e}, {:.2e}, {:.2e})".format(
                avg_loss, loss_c, loss_ox, loss_oy, loss_sx, loss_sy
            ),
            "| Range: ({:.2e}, {:.2e})".format(
                torch.sigmoid(preds[:, 0, :, :].min()).item(),
                torch.sigmoid(preds[:, 0, :, :].max()).item(),
            ),
            "| T: {:4.2f}s".format(time.time() - start),
        )

        if args.checkpoint != 0 and epoch % args.checkpoint == 0:
            numpy.savetxt(loss_history_file, loss_history[0:epoch])
            print("NOTE: Loss history available at {}".format(loss_history_file))

        if args.eval_per != 0 and epoch % args.eval_per == 0:
            with torch.no_grad():
                dirname = "../results/eval_{}/".format(epoch)
                os.makedirs(dirname, exist_ok=True)

                model.eval()

                eval_data, eval_images = get_eval_data2(args.batch_size, args.device)
                eval_preds = model(eval_data)

                if eval_preds.is_cuda:
                    eval_preds = eval_preds.cpu().numpy()
                else:
                    eval_preds = eval_preds.numpy()

                for i, (p, im) in enumerate(zip(eval_preds, eval_images)):
                    draw_pred_grid(im, p)
                    im.save(dirname + "{}.png".format(i))

                print("NOTE: Eval result available at {}".format(dirname))

                model.train()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", default="cpu")
    parser.add_argument("-b", "--batch-size", type=int, default=16)
    parser.add_argument("-e", "--max-epoch", type=int, default=2)
    parser.add_argument("-v", "--eval-per", type=int, default=2)
    parser.add_argument("-c", "--checkpoint", type=int, default=100)

    args = parser.parse_args()
    args.device = torch.device(args.device)

    model = LineModel()
    train_line(model, args)


if __name__ == "__main__":
    main()
