import torch
from PIL import Image
import random
from torchvision import transforms
import csv


def get_train_data(res, n_batch, n_anchor, n_grid):
    samples = random.sample(range(600), n_batch)
    jpg_files = [f"data_train/{s:03d}.jpg" for s in samples]
    txt_files = [f"data_train/{s:03d}.txt" for s in samples]

    # convert jpg files to NCWH
    images = [Image.open(file).convert("L") for file in jpg_files]
    ratio_x = [res / im.width for im in images]
    ratio_y = [res / im.height for im in images]
    transform = transforms.Compose(
        [transforms.Resize((res, res), Image.BICUBIC), transforms.ToTensor()]
    )
    images = torch.stack(list(map(transform, images)), dim=0)

    # convert txt files to NCWH
    grid_res = res // n_grid
    labels = torch.zeros(n_batch, 5 * n_anchor, n_grid, n_grid)
    for i, (f, rx, ry) in enumerate(zip(txt_files, ratio_x, ratio_y)):
        labels[i, :, :, :] = txt_to_tensor(f, rx, ry, n_anchor, n_grid, grid_res)

    return images, labels


def txt_to_tensor(txt_file, ratio_x, ratio_y, n_anchor, n_grid, grid_res):
    tensor = torch.zeros(5 * n_anchor, n_grid, n_grid)
    with open(txt_file, "r", encoding="utf-8", newline="") as csv_file:
        for line in csv.reader(csv_file):
            l = [int(coor) for coor in line[0:8]]
            box = [
                (l[0] + l[4]) / 2 * ratio_x,  # center-x
                (l[1] + l[5]) / 2 * ratio_y,  # center-y
                (l[4] - l[0]) * ratio_x,  # w
                (l[5] - l[1]) * ratio_y,  # h
            ]
            gx = int(box[0] / grid_res)
            gy = int(box[1] / grid_res)
            for c in [x * 5 for x in range(n_anchor)]:
                if tensor[c, gx, gy] == 0:
                    tensor[c : c + 5, gx, gy] = torch.tensor([1] + box)
                    break

    return tensor


if __name__ == "__main__":
    pass
