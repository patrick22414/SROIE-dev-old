import time

import numpy
import torch

RESO_H = 768
RESO_W = RESO_H // 2
GRID_H = 16
GRID_W = RESO_W // 5

# legacy globals
RESO_H = RESO_H
G_RESO = GRID_H


class LineModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.line_detector = torch.nn.Sequential(
            # 240 x 48 x 24
            torch.nn.Conv2d(240, 240, (1, 3), groups=3),
            torch.nn.BatchNorm2d(240),
            torch.nn.LeakyReLU(inplace=True),
            # 240 x 48 x 22
            torch.nn.Conv2d(240, 240, (1, 3), groups=3),
            torch.nn.BatchNorm2d(240),
            torch.nn.LeakyReLU(inplace=True),
            # 240 x 48 x 20
            torch.nn.Conv2d(240, 240, (1, RESO_H // 32 - 4), groups=3),
            # 240 x 48 x 1
            torch.nn.Conv2d(240, 3, 1, groups=3),
            # 3 x 48 x 1
        )

    def forward(self, input):
        features = self.feature_extractor(input)

        lines = self.line_detector(features).squeeze(dim=3)
        conf = lines[:, 0, :] if self.training else torch.sigmoid(lines[:, 0, :])
        offset = torch.nn.functional.softsign(lines[:, 1, :])
        scale = torch.nn.functional.softplus(lines[:, 2, :])

        return torch.stack([conf, offset, scale], dim=2)


class GridModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.grid_detector = torch.nn.Sequential(
            # 240 x 50 x 25
            torch.nn.Conv2d(240, 240, 3, padding=(1, 0), groups=5),
            torch.nn.BatchNorm2d(240),
            torch.nn.LeakyReLU(inplace=True),
            # 240 x 50 x 23
            torch.nn.Conv2d(240, 240, (1, 3), groups=5),
            torch.nn.BatchNorm2d(240),
            torch.nn.LeakyReLU(inplace=True),
            # 240 x 50 x 21
            torch.nn.Conv2d(240, 240, (1, 3), groups=5),
            torch.nn.BatchNorm2d(240),
            torch.nn.LeakyReLU(inplace=True),
            # 240 x 50 x 19
            torch.nn.Conv2d(240, 240, (1, 3), groups=5),
            torch.nn.BatchNorm2d(240),
            torch.nn.LeakyReLU(inplace=True),
            # 240 x 50 x 17
            torch.nn.Conv2d(240, 240, (1, 3), groups=5),
            torch.nn.BatchNorm2d(240),
            torch.nn.LeakyReLU(inplace=True),
            # 240 x 50 x 15
            torch.nn.Conv2d(240, 240, (1, 3), stride=(1, 3), groups=5),
            # 240 x 50 x 5
            torch.nn.Conv2d(240, 5, 1, groups=5),
            # 5 x 50 x 5
        )

    def forward(self, input):
        features = self.feature_extractor(input)

        grids = self.grid_detector(features)
        conf = grids[:, 0, :, :] if self.training else torch.sigmoid(grids[:, 0, :, :])
        offset_x = torch.nn.functional.softsign(grids[:, 1, :, :])
        offset_y = torch.nn.functional.softsign(grids[:, 2, :, :])
        scale_x = torch.nn.functional.softplus(grids[:, 3, :, :])
        scale_y = torch.nn.functional.softplus(grids[:, 4, :, :])

        return torch.stack([conf, offset_x, offset_y, scale_x, scale_y], dim=1)


class FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = torch.nn.Sequential(
            # 3 x 800 x 400
            torch.nn.Conv2d(3, 15, 3, padding=1),
            torch.nn.BatchNorm2d(15),
            torch.nn.LeakyReLU(inplace=True),
            # 15 x 800 x 400
            torch.nn.Conv2d(15, 15, 3, padding=1),
            torch.nn.BatchNorm2d(15),
            torch.nn.LeakyReLU(inplace=True),
            # 15 x 800 x 400
            torch.nn.Conv2d(15, 30, 3, stride=2, padding=1),
            torch.nn.BatchNorm2d(30),
            torch.nn.LeakyReLU(inplace=True),
            # 30 x 400 x 200
            Residual(30),
            # Inception(30),
            torch.nn.Conv2d(30, 60, 3, stride=2, padding=1),
            torch.nn.BatchNorm2d(60),
            torch.nn.LeakyReLU(inplace=True),
            # 60 x 200 x 100
            Residual(60),
            # Inception(60),
            torch.nn.Conv2d(60, 120, 3, stride=2, padding=1),
            torch.nn.BatchNorm2d(120),
            torch.nn.LeakyReLU(inplace=True),
            # 120 x 100 x 50
            Residual(120),
            # Inception(120),
            torch.nn.Conv2d(120, 240, 3, stride=2, padding=1),
            torch.nn.BatchNorm2d(240),
            torch.nn.LeakyReLU(inplace=True),
            # 240 x 50 x 25
        )

    def forward(self, input):
        return self.feature_extractor(input)


class Inception(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        half_chan = in_channels // 2

        self.inception_body = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.MaxPool2d(3, stride=1, padding=1),
                    torch.nn.Conv2d(in_channels, half_chan, 1),
                    torch.nn.BatchNorm2d(half_chan),
                    torch.nn.LeakyReLU(inplace=True),
                ),
                torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels, half_chan, 1),
                    torch.nn.BatchNorm2d(half_chan),
                    torch.nn.LeakyReLU(inplace=True),
                    torch.nn.Conv2d(half_chan, half_chan, 3, padding=1, groups=half_chan),
                    torch.nn.BatchNorm2d(half_chan),
                    torch.nn.LeakyReLU(inplace=True),
                ),
                torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels, half_chan, 1),
                    torch.nn.BatchNorm2d(half_chan),
                    torch.nn.LeakyReLU(inplace=True),
                    torch.nn.Conv2d(half_chan, half_chan, (1, 3), padding=(0, 1)),
                    torch.nn.Conv2d(half_chan, half_chan, (3, 1), padding=(1, 0)),
                    torch.nn.BatchNorm2d(half_chan),
                    torch.nn.LeakyReLU(inplace=True),
                ),
                torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels, half_chan, 1),
                    torch.nn.BatchNorm2d(half_chan),
                    torch.nn.LeakyReLU(inplace=True),
                    torch.nn.Conv2d(half_chan, half_chan, (1, 5), padding=(0, 2)),
                    torch.nn.Conv2d(half_chan, half_chan, (5, 1), padding=(2, 0)),
                    torch.nn.BatchNorm2d(half_chan),
                    torch.nn.LeakyReLU(inplace=True),
                ),
            ]
        )

        self.inception_tail = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels * 2, in_channels, 1),
            torch.nn.BatchNorm2d(in_channels),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Conv2d(in_channels, in_channels, 3, padding=1),
            torch.nn.BatchNorm2d(in_channels),
            torch.nn.LeakyReLU(inplace=True),
        )

    def forward(self, input):
        body_out = torch.cat([mod(input) for mod in self.inception_body], dim=1)
        tail_out = self.inception_tail(body_out)

        return tail_out


class Residual(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv_0 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, in_channels, 3, padding=1),
            torch.nn.BatchNorm2d(in_channels),
            torch.nn.LeakyReLU(inplace=True),
        )
        self.conv_1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, in_channels, 3, padding=1),
            torch.nn.BatchNorm2d(in_channels),
            torch.nn.LeakyReLU(inplace=True),
        )

    def forward(self, input):
        output_0 = self.conv_0(input) + input
        output_1 = self.conv_1(output_0) + output_0
        return output_1


if __name__ == "__main__":
    numpy.set_printoptions(precision=2, suppress=True)

    model = GridModel()

    data = torch.randn(2, 3, RESO_H, RESO_W)

    start = time.time()
    pred = model(data)
    print(pred.detach().numpy())
    print("T:", time.time() - start)
    print(pred.size())
    print(pred.max().item(), pred.min().item())
