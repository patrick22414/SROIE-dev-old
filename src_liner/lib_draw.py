import numpy
from PIL import Image, ImageDraw

from lib_model import G_RESO, H_RESO


def draw_prediction(image, pred, threshold=None):
    draw = ImageDraw.Draw(image)

    # pred = numpy.array(pred)  # shape == (512, 3)

    conf = pred[:, 0]
    center = pred[:, 1] * G_RESO + numpy.arange(G_RESO / 2, H_RESO, G_RESO)
    height = pred[:, 2] * G_RESO

    if threshold == None:
        mask = conf > 0.8 * numpy.max(conf)
    else:
        mask = conf > threshold

    center = center[mask]
    height = height[mask]
    y0 = center - height / 2
    y1 = center + height / 2

    for u, v in zip(y0, y1):
        draw.line([0, u, image.width, u], fill="violet", width=1)
        draw.line([0, v, image.width, v], fill="purple", width=2)
