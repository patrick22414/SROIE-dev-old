from PIL import Image, ImageDraw
import glob
import os
from csv import reader


def labelling():
    filenames = [os.path.splitext(f)[0] for f in glob.glob("../data_train/*.jpg")]
    jpg_files = [f + ".jpg" for f in filenames]
    txt_files = [f + ".txt" for f in filenames]

    for jpg, txt in zip(jpg_files, txt_files):
        image = Image.open(jpg).convert("L").convert("RGB")
        draw = ImageDraw.Draw(image)

        with open(txt, "r", encoding="utf-8", newline="") as csv:
            for line in reader(csv):
                rectangle = (
                    int(line[0]),
                    int(line[1]),
                    int(line[4]),
                    int(line[5]),
                )
                draw.rectangle(rectangle, outline="magenta")

        image.save("../label/" + os.path.basename(jpg) + ".jpg")
        print(jpg)


def labelling_with_grid():
    filenames = [os.path.splitext(f)[0] for f in glob.glob("../data_train/*.jpg")]
    jpg_files = [f + ".jpg" for f in filenames]
    txt_files = [f + ".txt" for f in filenames]

    for jpg, txt in zip(jpg_files, txt_files):
        image = Image.open(jpg).convert("L").convert("RGB")
        draw = ImageDraw.Draw(image)

        for x in range(0, image.width, image.width // 32):
            draw.line([x, 0, x, image.height], fill="skyblue", width=1)
        for y in range(0, image.height, image.width // 32):
            draw.line([0, y, image.width, y], fill="skyblue", width=1)

        with open(txt, "r", encoding="utf-8", newline="") as csv:
            for line in reader(csv):
                rectangle = (
                    int(line[0]),
                    int(line[1]),
                    int(line[4]),
                    int(line[5]),
                )
                draw.rectangle(rectangle, outline="magenta")

        image.save("../label_with_grid/" + os.path.basename(jpg) + ".jpg")
        print(jpg)

if __name__ == "__main__":
    labelling_with_grid()
