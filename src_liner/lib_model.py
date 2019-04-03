import torch
import time


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
            torch.nn.Conv2d(12, 12, 3, stride=(1, 2), padding=1),
            torch.nn.BatchNorm2d(12),
            torch.nn.LeakyReLU(inplace=True),
            # 12 x 512 x 128
            torch.nn.Conv2d(12, 12, (1, 3), padding=(0, 1)),
            torch.nn.Conv2d(12, 12, (7, 1), padding=(3, 0)),
            torch.nn.BatchNorm2d(12),
            torch.nn.LeakyReLU(inplace=True),
            #
            torch.nn.Conv2d(12, 12, (1, 3), padding=(0, 1)),
            torch.nn.Conv2d(12, 12, (7, 1), padding=(3, 0)),
            torch.nn.BatchNorm2d(12),
            torch.nn.LeakyReLU(inplace=True),
            #
            torch.nn.Conv2d(12, 24, 3, stride=(1, 2), padding=1),
            torch.nn.BatchNorm2d(24),
            torch.nn.LeakyReLU(inplace=True),
            # 24 x 512 x 64
            torch.nn.Conv2d(24, 24, (1, 3), padding=(0, 1)),
            torch.nn.Conv2d(24, 24, (7, 1), padding=(3, 0)),
            torch.nn.BatchNorm2d(24),
            torch.nn.LeakyReLU(inplace=True),
            #
            torch.nn.Conv2d(24, 24, (1, 3), padding=(0, 1)),
            torch.nn.Conv2d(24, 24, (7, 1), padding=(3, 0)),
            torch.nn.BatchNorm2d(24),
            torch.nn.LeakyReLU(inplace=True),
            #
            torch.nn.Conv2d(24, 48, 3, stride=(1, 2), padding=1),
            torch.nn.BatchNorm2d(48),
            torch.nn.LeakyReLU(inplace=True),
            # 48 x 512 x 32
            torch.nn.Conv2d(48, 48, (1, 3), padding=(0, 1)),
            torch.nn.Conv2d(48, 48, (7, 1), padding=(3, 0)),
            torch.nn.BatchNorm2d(48),
            torch.nn.LeakyReLU(inplace=True),
            #
            torch.nn.Conv2d(48, 48, (1, 3), padding=(0, 1)),
            torch.nn.Conv2d(48, 48, (7, 1), padding=(3, 0)),
            torch.nn.BatchNorm2d(48),
            torch.nn.LeakyReLU(inplace=True),
            #
            torch.nn.Conv2d(48, 96, 3, stride=(1, 2), padding=1),
            torch.nn.BatchNorm2d(96),
            torch.nn.LeakyReLU(inplace=True),
            # 96 x 512 x 16
            # Option 1:
            # torch.nn.Conv2d(96, 96, (1, 16), groups=3),
            # torch.nn.Conv2d(96, 3, 1, groups=3),
            # Option 2:
            torch.nn.Conv2d(96, 108, (1, 3), groups=3),
            torch.nn.BatchNorm2d(108),
            torch.nn.LeakyReLU(inplace=True),
            # 108 x 512 x 14
            torch.nn.Conv2d(108, 120, (1, 3), groups=3),
            torch.nn.BatchNorm2d(120),
            torch.nn.LeakyReLU(inplace=True),
            # 120 x 512 x 12
            torch.nn.Conv2d(120, 132, (1, 3), groups=3),
            torch.nn.BatchNorm2d(132),
            torch.nn.LeakyReLU(inplace=True),
            # 132 x 512 x 10
            torch.nn.Conv2d(132, 144, (1, 3), groups=3),
            torch.nn.BatchNorm2d(144),
            torch.nn.LeakyReLU(inplace=True),
            # 144 x 512 x 8
            torch.nn.Conv2d(144, 144, (1, 8), groups=3),
            torch.nn.Conv2d(144, 3, 1, groups=3),
        )

    def forward(self, input: torch.Tensor):
        features = self.network(input).squeeze()
        conf = torch.sigmoid(features[:, 0, :])
        offset = features[:, 1, :]
        scale = torch.exp(features[:, 2, :])
        return torch.stack([conf, offset, scale], dim=2)


if __name__ == "__main__":
    model = Model()

    inpt = torch.randn(4, 3, 512, 256)

    start = time.time()
    oupt = model(inpt)
    print(oupt)
    print("T:", time.time() - start)
    print(oupt.size())
    print(oupt.max().item(), oupt.min().item())
