import numpy as np
import matplotlib.pyplot as plt

print("This is problem 1 week4")
#TODO: Implement explicit Euler method to solve:
# y′(t) = 3y(t) − y**2(t), y(0) = y0
def y_analytic(t):
    c = (d/initial_condition)-1
    return d*(np.exp(d*t)/(c+np.exp(d*t)))


def euler_explicit(initial_condition, t, f):
    y = np.zeros(len(t))

    y[0] = initial_condition
    for n in  range(len(t)-1):
        h_n = t[n+1] - t[n]
        y[n+1] = y[n] + h_n*f(t[n],y[n])
    return y

d = 3
initial_condition = 1
initial_time = 0
end_time = 5
t = np.linspace(0,end_time,100) # equidistant time grid

def f(t, y, d=3):
    return d*y - y**2

y_euler = euler_explicit(initial_condition, t, f)

print(y_euler)

plt.figure(1)
plt.plot(t, y_euler, 'r', t, y_analytic(t), '--b')
plt.legend(['Explicit Euler', 'Analytic Soln'])
plt.show()

# t_end = 1
# t_start = 0
# time = np.linspace(t_start, t_end, 100)
# i = 0
# step_size_h = time[i+1] - time[i]
#
# x = [0]
#
# for j in range(1,len(time)):
#     # print(i)
#     x.append(x[0] + j*step_size_h)
# # print(x)
#
# # here instead of x we have t.
# y = [2] * len(time)
# for i in range(1, len(time)):
#     print(i)
#     print((y[i] + step_size_h * (3*y[i]) - y[i]**2))
#     y.append((y[i] + step_size_h * (3*y[i]) - y[i]**2))
# print(y)
#
#
# # y[i+1] = y[i] + h*f(x_i,y_i)
# # f(x_i, y_i) = 3y - y^2
# # x[0] = 0; y[0] = y0; f(x_0,y_0) = 3y0 - y0^2
# # print(time)
# # print(step_size_h)
# # print("")
# # for t in time:
# #     print(t)