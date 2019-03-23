from PIL import Image
import csv
import random
import torch
from torchvision import transforms

RESOLUTION = 320

TRAIN_IMAGE = [None] * 500
TRAIN_LABEL = [None] * 500

N_ANCHORS = 3
N_GRID = 20
GRID_SIZE = int(RESOLUTION / N_GRID)
# Total number of boxes predicted per image: 20*20*3 = 1200

ANCHORS = torch.tensor(
    [[GRID_SIZE, GRID_SIZE], [GRID_SIZE, 2 * GRID_SIZE], [GRID_SIZE, 4 * GRID_SIZE]]
)

transform = transforms.Compose(
    [transforms.Resize((RESOLUTION, RESOLUTION), Image.BICUBIC), transforms.ToTensor()]
)

for i in range(500):
    # prepare TRAIN_IMAGE
    jpg_file = f"data_train/{i:03d}.jpg"
    image = Image.open(jpg_file).convert("L")
    ratio_x = RESOLUTION / image.width
    ratio_y = RESOLUTION / image.height

    TRAIN_IMAGE[i] = transform(image)

    # prepare TRAIN_LABEL
    txt_file = f"data_train/{i:03d}.txt"
    label_tensor = torch.zeros(5 * N_ANCHORS, N_GRID, N_GRID)
    with open(txt_file, "r", encoding="utf-8", newline="") as csv_file:
        for l in csv.reader(csv_file):
            box = [
                (int(l[0]) + int(l[4])) / 2 * ratio_x,  # center-x
                (int(l[1]) + int(l[5])) / 2 * ratio_y,  # center-y
                (int(l[4]) - int(l[0])) * ratio_x,  # w
                (int(l[5]) - int(l[1])) * ratio_y,  # h
            ]
            gx = int(box[0] // GRID_SIZE)
            gy = int(box[1] // GRID_SIZE)
            for c in range(N_ANCHORS):
                if label_tensor[c * 5, gx, gy] == 0:
                    break
            else:
                continue
            label_tensor[c : c + 5, gx, gy] = torch.tensor([1] + box)

    TRAIN_LABEL[i] = label_tensor


def gen_data_batch(n, dataset="TRAIN"):
    samples = random.sample(range(len(TRAIN_IMAGE)), n)
    images = torch.stack([TRAIN_IMAGE[i] for i in samples], dim=0)
    labels = torch.stack([TRAIN_LABEL[i] for i in samples], dim=0)
    return images, labels


class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.network = torch.nn.Sequential(
            torch.nn.Conv2d(1, 4, 3, padding=1),  # -> 4 x 320 x 320
            torch.nn.BatchNorm2d(4),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(4, 8, 3, padding=1),  # -> 8 x 160 x 160
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(8, 16, 3, padding=1),  # -> 16 x 80 x 80
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(16, 32, 3, padding=1),  # -> 32 x 40 x 40
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(32, 15, 3, padding=1),  # -> 15 x 20 x 20
            torch.nn.Sigmoid()
        )

        self.grid_x_offset = torch.stack([torch.arange(0, float(N_GRID))] * N_GRID, dim=0).mul(GRID_SIZE)
        self.grid_y_offset = torch.stack([torch.arange(0, float(N_GRID))] * N_GRID, dim=1).mul(GRID_SIZE)

    def forward(self, inpt):
        return self.network(inpt)


model = MyModel()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

avg_loss = 0
for epoch in range(1000):
    images, labels = gen_data_batch(20)
    oupt = model.forward(images)

    loss = criterion(oupt, labels)
    loss.backward()
    optimizer.step()

    avg_loss = 0.9 * avg_loss + loss.item()
    if epoch % 10 == 0:
        print(avg_loss)
