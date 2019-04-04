import torch
import time

H_RESO = 800  # height resolution
G_RESO = 16  # grid resolution


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_extractor = torch.nn.Sequential(
            # 3 x 800 x 400
            torch.nn.Conv2d(3, 6, 3, padding=1),
            torch.nn.BatchNorm2d(6),
            torch.nn.LeakyReLU(inplace=True),
            # 6 x 800 x 400
            torch.nn.Conv2d(6, 12, 3, padding=1),
            torch.nn.BatchNorm2d(12),
            torch.nn.LeakyReLU(inplace=True),
            # 12 x 800 x 400
            torch.nn.Conv2d(12, 24, 3, stride=2, padding=1),
            torch.nn.BatchNorm2d(24),
            torch.nn.LeakyReLU(inplace=True),
            # 24 x 400 x 200
            torch.nn.Conv2d(24, 24, 3, padding=1),
            torch.nn.BatchNorm2d(24),
            torch.nn.LeakyReLU(inplace=True),
            #
            torch.nn.Conv2d(24, 24, 3, padding=1),
            torch.nn.BatchNorm2d(24),
            torch.nn.LeakyReLU(inplace=True),
            #
            torch.nn.Conv2d(24, 48, 3, stride=2, padding=1),
            torch.nn.BatchNorm2d(48),
            torch.nn.LeakyReLU(inplace=True),
            # 48 x 200 x 100
            torch.nn.Conv2d(48, 48, 3, padding=1),
            torch.nn.BatchNorm2d(48),
            torch.nn.LeakyReLU(inplace=True),
            #
            torch.nn.Conv2d(48, 48, 3, padding=1),
            torch.nn.BatchNorm2d(48),
            torch.nn.LeakyReLU(inplace=True),
            #
            torch.nn.Conv2d(48, 96, 3, stride=2, padding=1),
            torch.nn.BatchNorm2d(96),
            torch.nn.LeakyReLU(inplace=True),
            # 96 x 100 x 50
            torch.nn.Conv2d(96, 96, 3, padding=1),
            torch.nn.BatchNorm2d(96),
            torch.nn.LeakyReLU(inplace=True),
            #
            torch.nn.Conv2d(96, 96, 3, padding=1),
            torch.nn.BatchNorm2d(96),
            torch.nn.LeakyReLU(inplace=True),
            #
            torch.nn.Conv2d(96, 192, 3, stride=2, padding=1),
            torch.nn.BatchNorm2d(192),
            torch.nn.LeakyReLU(inplace=True),
        )

        self.line_detector = torch.nn.Sequential(
            # 192 x 50 x 25
            torch.nn.Conv2d(192, 216, (1, 3), groups=3),
            torch.nn.BatchNorm2d(216),
            torch.nn.LeakyReLU(inplace=True),
            # 216 x 50 x 23
            torch.nn.Conv2d(216, 240, (1, 3), groups=3),
            torch.nn.BatchNorm2d(240),
            torch.nn.LeakyReLU(inplace=True),
            # 240 x 50 x 21
            torch.nn.Conv2d(240, 240, (1, H_RESO // 32 - 4), groups=3),
            torch.nn.Conv2d(240, 3, 1, groups=3),
        )

    def forward(self, input):
        features = self.feature_extractor(input)
        lines = self.line_detector(features).squeeze(dim=3)

        conf = lines[:, 0, :] if self.training else torch.sigmoid(lines[:, 0, :])

        offset = lines[:, 1, :]

        scale = torch.exp(lines[:, 2, :])

        return torch.stack([conf, offset, scale], dim=2)


if __name__ == "__main__":
    model = Model()

    data = torch.randn(2, 3, H_RESO, H_RESO // 2)

    start = time.time()
    pred = model(data)
    print(pred)
    print("T:", time.time() - start)
    print(pred.size())
    print(pred.max().item(), pred.min().item())
