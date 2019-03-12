from PIL import Image, ImageDraw
import csv
import torch

RESOLUTION = 320

DATA_BOX = []  # 600 tensors, each <n_boxes-by-4>
DATA_STR = []  # 600 lists


def txt_to_list(txt_file, ratio_x=1, ratio_y=1):
    with open(txt_file, "r", encoding="utf-8", newline="") as csv_file:
        ans = [
            [
                int(l[0]) * ratio_x,
                int(l[1]) * ratio_y,
                int(l[4]) * ratio_x,
                int(l[5]) * ratio_y,
            ]
            for l in csv.reader(csv_file)
        ]
    return ans


for i in range(10):
    jpg_file = f"data_train/{i:03d}.jpg"
    txt_file = f"data_train/{i:03d}.txt"

    # prepare image data
    image = Image.open(jpg_file).convert("L")
    ratio_x = RESOLUTION / image.width
    ratio_y = RESOLUTION / image.height
    image = image.resize((RESOLUTION, RESOLUTION), Image.BICUBIC)

    boxes = txt_to_list(txt_file, ratio_x, ratio_y)

    draw = ImageDraw.Draw(image)
    for box in boxes:
        draw.rectangle(box, outline=(0))

    image.save(f"data_resized/{i:03d}.png")
