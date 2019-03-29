import torch
import sys
import argparse


def my_function(x):
    t = [None]*3
    t[0] = x[0, :, :].add(10)
    t[1] = x[1, :, :].mul(0.5)
    t[2] = x[2, :, :].exp()
    return torch.stack(t, dim=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device')
    args = parser.parse_args()

    print(args.device)
