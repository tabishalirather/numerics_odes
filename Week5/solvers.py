print("Loading solvers.py")

import numpy as np

def improved_euler(f, ic, t):
    print("calling improved euler")
    y=np.zeros(len(t))
    y[0] = ic
    for n in range(len(t)-1):
        print("for running")
        h_n = t[n+1] - t[n]
        y_half  = y[n] + (h_n/2)*f(t[n], y[n])
        y[n+1] = y[n] + h_n*f(t[n] + h_n/2, y_half)

    return y


def explicit_euler(f, ic, t):
    print("calling explicit euler")
    y=np.zeros(len(t))
    y[0] = ic
    for n in range(len(t)-1):
        print("for running")
        h_n = t[n+1] - t[n]
        # y_half  = y[n] + (h_n/2)*f(t[n], y[n])
        y[n+1] = y[n] + h_n*f(t[n] + h_n, y[n])

    return y