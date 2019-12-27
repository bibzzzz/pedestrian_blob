import numpy
import pandas


def find_options(x, y):
    options = []
    if x == floor(x):
        options.append(1)
    else:
        options.append(0)

    if y == floor(y):
        options.append(1)
    else:
        options.append(0)
