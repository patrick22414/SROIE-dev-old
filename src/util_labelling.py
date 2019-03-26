import glob
import os
from dataclasses import dataclass
from PIL import Image, ImageDraw
import csv
import string


class Box:
    def __init__(self, rect: list, text: str):
        self.rect = list(map(int, rect))
        self.text = "".join(filter(lambda c: c in string.printable, text))


def file_to_boxes(filename):
    with open(filename, "r", encoding="utf-8", newline="") as csv_file:
        boxes = [Box([l[0], l[1], l[4], l[5]], l[8]) for l in csv.reader(csv_file)]
    return boxes


for i in range(1):
    jpg_file = f"train_data/{i:03d}.jpg"
    txt_file = f"train_data/{i:03d}.txt"
    boxes = file_to_boxes(txt_file)
    image = Image.open(jpg_file).convert("RGB")

    draw = ImageDraw.Draw(image)
    print(dir(draw))
    for box in boxes:
        draw.rectangle(box.rect, outline=(255, 0, 0))
        draw.text([box.rect[0] + 2, box.rect[1]], box.text, fill=(255, 0, 0))

    image.save(f"labelled/{i:03d}.png")

