import numpy
from PIL import Image, ImageDraw

from lib_model import RESO_H, RESO_W, GRID_H, GRID_W


def draw_pred_line(image, pred, threshold=None):
    draw = ImageDraw.Draw(image)

    # pred = numpy.array(pred)  # shape == (512, 3)

    conf = pred[:, 0]
    center = pred[:, 1] * GRID_H + numpy.arange(GRID_H / 2, RESO_H, GRID_H)
    height = pred[:, 2] * GRID_H

    if threshold == None:
        mask = conf > 0.8 * numpy.max(conf)
    else:
        mask = conf > threshold

    center = center[mask]
    height = height[mask]
    y0 = (center - height / 2) / (RESO_H / image.height)
    y1 = (center + height / 2) / (RESO_H / image.height)

    for u, v in zip(y0, y1):
        draw.line([0, u, image.width, u], fill="orange", width=2)
        draw.line([0, v, image.width, v], fill="red", width=2)


def draw_pred_grid(image, pred, threshold=None):
    # pred.shape == (5, RESO_H // GRID_H, RESO_W // GRID_W) or (5, 50, 5)
    draw = ImageDraw.Draw(image)

    conf = pred[0]
    if threshold == None:
        mask = conf > 0.5 * numpy.max(conf)
    else:
        mask = conf > threshold

    center_x = numpy.stack([numpy.arange(GRID_W // 2, RESO_W, GRID_W)] * (RESO_H // GRID_H), axis=0)
    center_y = numpy.stack([numpy.arange(GRID_H // 2, RESO_H, GRID_H)] * (RESO_W // GRID_W), axis=1)

    center_x = (pred[1] * GRID_W + center_x)[mask]
    center_y = (pred[2] * GRID_H + center_y)[mask]

    w_half = pred[3][mask] * GRID_W / 2
    h_half = pred[4][mask] * GRID_H / 2

    x0 = (center_x - w_half) / (RESO_W / image.width)
    y0 = (center_y - h_half) / (RESO_H / image.height)
    x1 = (center_x + w_half) / (RESO_W / image.width)
    y1 = (center_y + h_half) / (RESO_H / image.height)

    for rect in zip(x0, y0, x1, y1):
        draw.rectangle(rect, outline="magenta")
