import torch

GRID_RESO = 16


class MyModel(torch.nn.Module):
    def __init__(self, reso, anchors, device):
        super().__init__()

        self.reso = reso
        self.anchors = anchors
        self.n_anchor = len(anchors)
        self.n_grid = int(self.reso / GRID_RESO)

        self.anchor_tensor_w = torch.stack(
            [torch.full((self.n_grid, self.n_grid), a[0], device=device) for a in self.anchors], dim=0
        )
        self.anchor_tensor_h = torch.stack(
            [torch.full((self.n_grid, self.n_grid), a[1], device=device) for a in self.anchors], dim=0
        )

        self.grid_offset_x = torch.arange(float(self.n_grid), device=device).repeat(self.n_grid, 1).mul(GRID_RESO)
        self.grid_offset_y = self.grid_offset_x.t()

        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Conv2d(1, 4, 3, padding=1),
            torch.nn.BatchNorm2d(4),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            #
            torch.nn.Conv2d(4, 8, 3, padding=1),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            #
            torch.nn.Conv2d(8, 16, 3, padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            #
            torch.nn.Conv2d(16, 32, 3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            #
            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
        ).to(device=device)

        self.conv_c = torch.nn.Conv2d(64, self.n_anchor, 3, padding=1).to(device=device)
        self.conv_x = torch.nn.Conv2d(64, self.n_anchor, 3, padding=1).to(device=device)
        self.conv_y = torch.nn.Conv2d(64, self.n_anchor, 3, padding=1).to(device=device)
        self.conv_w = torch.nn.Conv2d(64, self.n_anchor, 3, padding=1).to(device=device)
        self.conv_h = torch.nn.Conv2d(64, self.n_anchor, 3, padding=1).to(device=device)

    def forward(self, inpt):
        feature = self.feature_extractor(inpt)

        c = torch.sigmoid(self.conv_c(feature))

        x = torch.sigmoid(self.conv_x(feature)).mul(GRID_RESO).add(self.grid_offset_x)

        y = torch.sigmoid(self.conv_y(feature)).mul(GRID_RESO).add(self.grid_offset_y)

        w = torch.tanh(self.conv_w(feature)).exp().mul(self.anchor_tensor_w)

        h = torch.tanh(self.conv_h(feature)).exp().mul(self.anchor_tensor_h)

        return c, x, y, w, h


if __name__ == "__main__":
    pass
