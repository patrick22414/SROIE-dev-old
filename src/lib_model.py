import torch


class MyModel(torch.nn.Module):
    def __init__(self, res, anchor):
        super().__init__()

        self.res = res
        self.anchor = anchor
        self.n_anchor = len(anchor)
        self.scale = 16
        self.n_grid = int(self.res / self.scale)

        self.network = torch.nn.Sequential(
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
        oupt_0 = self.network(inpt)
        # oupt_0 = torch.ones([2, 15, 5, 5])  # DEBUG
        oupt_1 = torch.zeros_like(oupt_0, requires_grad=True)
        for i, a in enumerate(self.anchor):
            oupt_1[:, i * 5, :, :] = oupt_0[:, i * 5, :, :]
            oupt_1[:, i * 5 + 1, :, :] = (
                oupt_0[:, i * 5 + 1, :, :].mul(self.scale).add(self.grid_x_offset)
            )
            oupt_1[:, i * 5 + 2, :, :] = (
                oupt_0[:, i * 5 + 2, :, :].mul(self.scale).add(self.grid_y_offset)
            )
            oupt_1[:, i * 5 + 3, :, :] = (
                oupt_0[:, i * 5 + 3, :, :].mul(2).add(-1).exp().mul(a[0])
            )
            oupt_1[:, i * 5 + 4, :, :] = (
                oupt_0[:, i * 5 + 4, :, :].mul(2).add(-1).exp().mul(a[1])
            )

        return oupt_1


if __name__ == "__main__":
    model = MyModel(80, [(16, 16), (10, 10), (16, 32)])
    print(model.forward(torch.zeros(2, 1, 80, 80)))

