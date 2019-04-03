import torch
import time

H_RESO = 512  # height resolution
G_RESO = 16  # grid resolution


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.network = torch.nn.Sequential(
            # 3 x 512 x 256
            torch.nn.Conv2d(3, 6, 3, padding=1),
            torch.nn.BatchNorm2d(6),
            torch.nn.LeakyReLU(inplace=True),
            # 6 x 512 x 256
            torch.nn.Conv2d(6, 12, 3, padding=1),
            torch.nn.BatchNorm2d(12),
            torch.nn.LeakyReLU(inplace=True),
            # 12 x 512 x 256
            torch.nn.Conv2d(12, 24, 3, stride=2, padding=1),
            torch.nn.BatchNorm2d(24),
            torch.nn.LeakyReLU(inplace=True),
            # 24 x 256 x 128
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
            # 48 x 128 x 64
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
            # 96 x 64 x 32
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
            #
            # 192 x 32 x 16
            torch.nn.Conv2d(192, 216, (1, 3), groups=3),
            torch.nn.BatchNorm2d(216),
            torch.nn.LeakyReLU(inplace=True),
            # 216 x 32 x 14
            torch.nn.Conv2d(216, 240, (1, 3), groups=3),
            torch.nn.BatchNorm2d(240),
            torch.nn.LeakyReLU(inplace=True),
            # 240 x 32 x 12
            torch.nn.Conv2d(240, 240, (1, 12), groups=3),
            torch.nn.Conv2d(240, 3, 1, groups=3),
        )

    def forward(self, input: torch.Tensor):
        features = self.network(input).squeeze(dim=3)

        conf = features[:, 0, :] if self.training else torch.sigmoid(features[:, 0, :])

        offset = features[:, 1, :]

        scale = torch.exp(features[:, 2, :])

        return torch.stack([conf, offset, scale], dim=2)


if __name__ == "__main__":
    model = Model()

    data = torch.randn(4, 3, 512, 256)

    start = time.time()
    pred = model(data)
    print(pred)
    print("T:", time.time() - start)
    print(pred.size())
    print(pred.max().item(), pred.min().item())
