import torch


class Anchor:
    def __init__(self, w, h):
        self.w = w
        self.h = h


class BBox:
    def __init__(self, *args):
        if len(args) == 4:
            if isinstance(args, torch.Tensor):
                args = args.numpy()
            self.x = args[0]
            self.y = args[1]
            self.w = args[2]
            self.h = args[3]
        else:
            raise TypeError("BBox() must take 4 numbers or 1 list of 4 numbers")


if __name__ == "__main__":
    anchor = Anchor(1, 2)
    print(anchor.w)
