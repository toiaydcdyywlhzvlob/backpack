from torch import sum as torch_sum

NUMERICAL_STABILITY_CONSTANT = 1e-8


def flatten(list_of_lists):
    return list([item for sublist in list_of_lists for item in sublist])


def inner_product(xs, ys):
    return sum([torch_sum(x * y) for x, y in zip(xs, ys)])
