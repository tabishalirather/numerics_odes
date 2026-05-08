from fontTools.diff import color

print("week 5 problem 2")
import numpy as np
import matplotlib.pyplot as plt
from solvers import explicit_euler, improved_euler

x_start = 0.8
x_end = 1.8
ic = 5/6

def f(x,y):
    return y**2

def y_analytic(x):
    return 1/(2-x)

h = 0.05
x = np.linspace(x_start, x_end, int((x_end-x_start)/h))

y_euler_improved = improved_euler(f, ic, x)
y_euler_explicit = explicit_euler(f, ic, x)
y_analytic_sol = y_analytic(x)

plt.figure(1)
plt.plot(x, y_euler_improved,'b', x, y_euler_explicit, 'r', x, y_analytic_sol, 'g')
plt.legend(['improved', 'explicit', 'analytic'])
plt.ylabel('y')
plt.xlabel('x')
# plt.xlim([0,1.4])
# plt.ylim([y_euler_explicit[0], 4])
plt.show()


h = np.array([1e-1, 1e-2, 1e-3, 1e-4])
improved_euler_errs = np.zeros(len(h))
explicit_euler_errs = np.zeros(len(h))
for index in range(len(h)):
    x = np.linspace(x_start, x_end, int((x_end - x_start) / h[index]))
    y_euler_improved = improved_euler(f, ic, x)
    y_euler_explicit = explicit_euler(f, ic, x)
    y_analytic_sol = y_analytic(x)
    explicit_euler_errs[index] = np.max(np.abs(y_analytic_sol-y_euler_explicit))
    improved_euler_errs[index] = np.max(np.abs(y_analytic_sol - y_euler_improved))

plt.figure(2)
plt.loglog(h, explicit_euler_errs,'b', h, improved_euler_errs, 'r', h,h, '--b', h, h**2, '-.r')
plt.show()
plt.legend(['Explicit','improved', 'slope=1', 'slope=2'])




