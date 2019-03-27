import torch


def my_function(x):
    t = [None]*3
    t[0] = x[0, :, :].add(10)
    t[1] = x[1, :, :].mul(0.5)
    t[2] = x[2, :, :].exp()
    return torch.stack(t, dim=0)


if __name__ == "__main__":
    x = torch.rand(3, 2, 2, requires_grad=True)
    y = my_function(x)
    print(y)

    z = y.sum()
    print(z)

    z.backward()
    print(x.grad)
