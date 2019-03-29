import argparse
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", default="cpu")
    parser.add_argument("-r", "--resolution", type=int, default=320)
    parser.add_argument("-b", "--batch-size", type=int, default=10)

    args = parser.parse_args()

    print(args.device)
    print(args.resolution)
    print(args.batch_size)

    device = torch.device(args.device)

    print(device)
    print(torch.cuda.current_device())


if __name__ == "__main__":
    main()
