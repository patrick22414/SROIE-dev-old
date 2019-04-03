import argparse
import glob
import os
import time

import torch

from lib_data import get_eval_data, get_train_data
from lib_draw import draw_prediction
from lib_model import Model


def train(model, args):
    model.to(args.device)
    model.train()

    bce_loss = torch.nn.BCEWithLogitsLoss()
    mse_loss = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1000)

    avg_loss = 0

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

        print(
            "#{:04d} | Loss: {:4.2f} ({:4.2f}, {:4.2f}, {:4.2f}) | Range: ({:.2f}, {:.2f})".format(
                epoch,
                avg_loss,
                loss_c,
                loss_o,
                loss_s,
                torch.sigmoid(preds[:, :, 0].min()).item(),
                torch.sigmoid(preds[:, :, 0].max()).item(),
            ),
            "| T: {:4.2f}s".format(time.time() - start),
        )

        if args.eval_per != 0 and epoch % args.eval_per == 0:
            with torch.no_grad():
                dirname = "../result_liner/eval_{}/".format(epoch)
                os.makedirs(dirname, exist_ok=True)

                model.eval()

                eval_data, eval_images = get_eval_data(4, args.device)
                eval_preds = model(eval_data)
                for i, (pred, image) in enumerate(zip(eval_preds, eval_images)):
                    draw_prediction(image, pred)
                    image.save(dirname + "{}.jpg".format(i))

                print("NOTE: Eval result available at {}".format(dirname))

                model.train()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", default="cpu")
    parser.add_argument("-b", "--batch-size", type=int, default=16)
    parser.add_argument("-e", "--max-epoch", type=int, default=2)
    parser.add_argument("-v", "--eval-per", type=int, default=2)

    args = parser.parse_args()
    args.device = torch.device(args.device)

    model = Model()
    train(model, args)


if __name__ == "__main__":
    main()
