import random

import numpy as np
from numpy.matlib import rand
from scipy.integrate import odeint
import matplotlib.pyplot as plt

print("Printing from problem3.py")


# Given diff eqn is: y' = f(y), y(0) = y_0
# f(y) = y^c, y_0 = 0 and c in (0,1)4


def model(_y, _t, _c):
    # c = 0.2
    dy_dt = _y ** _c
    print(dy_dt)
    return dy_dt


def plt_with_odeint(c,t, initial_value_array):
    for value in initial_value_array:
        y = odeint(model, value, t, args=(c,))
        plt.plot(t, y, label=f"y_0 = {value}")
    plt.title(f"Problem 3: Solutions for c={c}")
    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.ylim(-c*10,c*(10**2))
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    # y_0 = 1e-20

    # model = model()
    t = np.linspace(0, 100, 200)
    initial_value_array = [0, 1e-5, 1e-10, 10e-15]
    c =random.random()
    print(f"c is: {c}")
    title = "Problem 3"
    plt_with_odeint(c, t, initial_value_array)

main()