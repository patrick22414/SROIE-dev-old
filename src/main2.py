import argparse
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", default="cpu")
    parser.add_argument("-r", "--resolution", type=int, default=320)
    parser.add_argument("-b", "--batch-size", type=int, default=10)

    args = parser.parse_args()

    args.device = torch.device(args.device)

    if args.device == torch.device('cpu'):
        run(args)
    else:
        run_cuda(args)


def run(args):
    print("running with cpu")


def run_cuda(args):
    print("running with cuda")
    with torch.cuda.device(args.device):
        print(torch.cuda.current_device())
        print(torch.zeros(2, 3).cuda())


if __name__ == "__main__":
    main()
