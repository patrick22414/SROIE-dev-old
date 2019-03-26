import torch
from PIL import Image
import random
from torchvision import transforms


def get_train_data(res, n_batch):
    samples = random.sample(range(600), n_batch)
    jpg_files = [f"data_train/{s:03d}.jpg" for s in samples]
    txt_files = [f"data_train/{s:03d}.txt" for s in samples]

    # convert jpg files to NCWH
    transform = transforms.Compose(
        [transforms.Resize((res, res), Image.BICUBIC), transforms.ToTensor()]
    )
    images = [Image.open(file).convert('L') for file in jpg_files]
    images = torch.stack([transform(im) for im in images], dim=0)

    return images


if __name__ == "__main__":
    pass
