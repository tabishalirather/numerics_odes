import numpy as np
print("Hello.py")

# implementing euler improved and Heuns Method via butcher tables .
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

def heun(f,ic, t):
    y = np.zeros(len(t))
    y[0] = ic

    for n in range(len(t)-1):
        h = t[n+1] - t[n]
        k1 = f(t[n],y[n])
        k2 = f(t[n]+h, y[n]+(h*k1))
        y[n+1] = y[n] + h/2 * (k1+k2)
    return y

