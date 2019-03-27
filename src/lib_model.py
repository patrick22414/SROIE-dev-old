import torch


class MyModel(torch.nn.Module):
    def __init__(self, res, anchor):
        super().__init__()

        self.res = res
        self.anchor = anchor
        self.n_anchor = len(anchor)
        self.scale = 16
        self.n_grid = int(self.res / self.scale)

        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Conv2d(1, 4, 3, padding=1),
            torch.nn.BatchNorm2d(4),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(4, 8, 3, padding=1),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(8, 16, 3, padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, 3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 5 * self.n_anchor, 3, padding=1),
            torch.nn.Sigmoid(),
        )

        self.grid_x_offset = (
            torch.arange(float(self.n_grid)).repeat(self.n_grid, 1).mul(self.scale)
        )
        self.grid_y_offset = (
            torch.arange(float(self.n_grid)).repeat(self.n_grid, 1).t().mul(self.scale)
        )

    def forward(self, inpt):
        oupt_0 = self.feature_extractor(inpt)
        oupt_1 = self.weigh_anchor(oupt_0)

        return oupt_1

    def weigh_anchor(self, inpt):
        oupt = [None] * self.n_anchor
        for i, a in enumerate(self.anchor):
            oupt[i] = torch.stack(
                [
                    inpt[:, i * 5, :, :],
                    inpt[:, i * 5 + 1, :, :].mul(self.scale).add(self.grid_x_offset),
                    inpt[:, i * 5 + 2, :, :].mul(self.scale).add(self.grid_y_offset),
                    inpt[:, i * 5 + 3, :, :].mul(2).add(-1).exp().mul(a[0]),
                    inpt[:, i * 5 + 4, :, :].mul(2).add(-1).exp().mul(a[1]),
                ],
                dim=1,
            )
        return torch.stack(oupt, dim=1)


if __name__ == "__main__":
    pass
