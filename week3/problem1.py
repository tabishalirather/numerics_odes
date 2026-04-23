print("Printing from problem1.py")
from problem3 import plt_with_odeint
import random

import numpy as np
from numpy.matlib import rand
from scipy.integrate import odeint
import matplotlib.pyplot as plt


# Given diff eqn is: y' = f(y), y(0) = y_0
# f(y) = y^c, y_0 = 0 and c in (0,1)4

def model(_y, _t, k):
    # c = 0.2
    dy_dt = k * _y
    print(dy_dt)
    return dy_dt


def plt_with_odeint(c, t, initial_value_array):
    k = c
    for value in initial_value_array:
        y = odeint(model, value, t, args=(k,))
        plt.plot(t, y, label=f"y_0 = {value}")
    plt.title(f"Problem 1: Solutions for c={k}")
    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.ylim(-k * 100, k * (100))
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    # y_0 = 1e-20
    initial_value_array = [0, 2]
    c = 10
    print(f"c is: {c}")
    t = np.linspace(0, 100, 200)
    plt_with_odeint(c, t, initial_value_array)


main()
