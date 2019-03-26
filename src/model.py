import torch


class MyModel(torch.nn.Module):
    def __init__(self, res, anchors):
        super().__init__()

        self.res = res
        self.anchors = anchors
        self.n_anchors = len(anchors)
        self.scale = 16
        self.n_grid = self.res // self.scale

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
            torch.nn.Conv2d(32, 5 * self.n_anchors, 3, padding=1),
            torch.nn.Sigmoid(),
        )

        self.grid_x_offset = (
            torch.arange(float(self.n_grid)).repeat(self.n_grid, 1).mul(self.scale)
        )
        self.grid_y_offset = (
            torch.arange(float(self.n_grid)).repeat(self.n_grid, 1).t().mul(self.scale)
        )

    def forward(self, inpt):
        oupt = self.network(inpt)
        oupt = torch.ones([2, 15, 5, 5]) # DEBUG
        for i, a in enumerate(self.anchors):
            oupt[:, i * 5 + 1, :, :].mul_(self.scale).add_(self.grid_x_offset)
            oupt[:, i * 5 + 2, :, :].mul_(self.scale).add_(self.grid_y_offset)
            oupt[:, i * 5 + 3, :, :].mul_(2).add_(-1).exp_().mul_(a[0])
            oupt[:, i * 5 + 4, :, :].mul_(2).add_(-1).exp_().mul_(a[1])

        return oupt


if __name__ == "__main__":
    model = MyModel(80, [(16, 16), (10, 10), (16, 32)])
    print(model.forward(torch.zeros(2, 1, 80, 80)))
    