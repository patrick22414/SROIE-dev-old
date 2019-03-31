import torch

GRID_RESO = 16


class MainModel(torch.nn.Module):
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
            BasicConv(3, 4, kernel_size=3, padding=1),
            Reduction(4),  # -> 8 x 256 x 256
            #
            InceptionResidual(8),
            Reduction(8),  # -> 16 x 128 x 128
            #
            InceptionResidual(16),
            Reduction(16),  # -> 32 x 64 x 64
            #
            InceptionResidual(32),
            Reduction(32),  # -> 64 x 32 x 32
            #
            BasicConv(64, 32, kernel_size=1, padding=0) # -> 32 x 32 x 32
        ).to(device=device)

        self.conv_c = torch.nn.Conv2d(32, self.n_anchor, 3, padding=1).to(device=device)
        self.conv_x = torch.nn.Conv2d(32, self.n_anchor, 3, padding=1).to(device=device)
        self.conv_y = torch.nn.Conv2d(32, self.n_anchor, 3, padding=1).to(device=device)
        self.conv_w = torch.nn.Conv2d(32, self.n_anchor, 3, padding=1).to(device=device)
        self.conv_h = torch.nn.Conv2d(32, self.n_anchor, 3, padding=1).to(device=device)

    def forward(self, inpt):
        feature = self.feature_extractor(inpt)

        # Do not apply sigmoid to confidence at training while using BCEWithLogitsLoss
        if self.training:
            c = self.conv_c(feature)
        else:
            c = torch.sigmoid(self.conv_c(feature))
            print("NOTE: In eval mode, confidence is computed with sigmoid")
            print(c[0, 0, 0, :])

        x = torch.sigmoid(self.conv_x(feature)).mul(GRID_RESO).add(self.grid_offset_x)

        y = torch.sigmoid(self.conv_y(feature)).mul(GRID_RESO).add(self.grid_offset_y)

        # w = torch.tanh(self.conv_w(feature)).exp().mul(self.anchor_tensor_w)
        w = self.conv_w(feature).exp().mul(self.anchor_tensor_w)

        # h = torch.tanh(self.conv_h(feature)).exp().mul(self.anchor_tensor_h)
        h = self.conv_h(feature).exp().mul(self.anchor_tensor_h)

        return c, x, y, w, h


class InceptionResidual(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        inner_channels = int(in_channels / 2)
        self.branch_1 = torch.nn.Sequential(
            torch.nn.MaxPool2d(3, 1, padding=1),
            BasicConv(in_channels, inner_channels, kernel_size=1, padding=0)
        )
        self.branch_3 = torch.nn.Sequential(
            BasicConv(in_channels, inner_channels, kernel_size=1, padding=0),
            BasicConv(inner_channels, inner_channels, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv(inner_channels, inner_channels, kernel_size=(3, 1), padding=(1, 0)),
        )
        self.branch_7 = torch.nn.Sequential(
            BasicConv(in_channels, inner_channels, kernel_size=1, padding=0),
            BasicConv(inner_channels, inner_channels, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv(inner_channels, inner_channels, kernel_size=(7, 1), padding=(3, 0)),
        )
        self.branch_out = BasicConv(inner_channels * 3, in_channels, kernel_size=1, padding=0)

    def forward(self, inpt):
        b1 = self.branch_1(inpt)
        b3 = self.branch_3(inpt)
        b7 = self.branch_7(inpt)
        cat = torch.cat([b1, b3, b7], dim=1)
        bo = self.branch_out(cat)
        oupt = torch.add(inpt, bo)
        return oupt


class BasicConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.sequential = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, inpt):
        return self.sequential(inpt)


class Reduction(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv_stride2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1),
            torch.nn.BatchNorm2d(in_channels),
            torch.nn.ReLU(inplace=True),
        )
        self.pool_stride2 = torch.nn.MaxPool2d(2, stride=2)

    def forward(self, inpt):
        return torch.cat([self.conv_stride2(inpt), self.pool_stride2(inpt)], dim=1)


if __name__ == "__main__":
    pass
