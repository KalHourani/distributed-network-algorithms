#!/usr/bin/env python3

from random import randint
import os


def generate_data(data_size, min_val=0, max_val=10**10):
    if not os.path.exists("inputs"):
        os.mkdir("inputs")
    with open("inputs/input_data_{0}.txt".format(data_size), "w") as file:
        for i in range(data_size):
            n = randint(min_val, max_val)
            file.write("{0}\n".format(n))


def main():
    for i in range(10, 21):
        data_size = 2 ** i
        generate_data(data_size)


if __name__ == "__main__":
    main()
