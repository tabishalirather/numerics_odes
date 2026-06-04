import matplotlib.pyplot as plt
import numpy as np
from butcher import improved_euler, heun
print("print from solver.py")

t_start = 0
t_end = 4
ic = 1

def f(t,y):
    return (1-(t*t)/2)*y

def analytic_solution(t):
    return np.exp(t-(t**3)/6)

h = 0.01
t= np.linspace(t_start, t_end, int((t_end-t_start)/h))

improved_euler_solution = improved_euler(f,ic,t)
heun_solution = heun(f,ic,t)
analytic_solution_model = analytic_solution(t)
plt.figure(1)
# plt.plot(t,analytic_solution,'k')

plt.plot(t,analytic_solution_model,'k',  t, improved_euler_solution, 'r', t, heun_solution, '--', 'g')
plt.legend(['Analytic', 'Improved Euler', 'Heun'])

h=np.logspace(-3,-1,10)  #starts from 10e-3 to 10e-1, log steps
err_euler_improved = np.zeros(len(h))
err_heun = np.zeros(len(h))

for i in range(len(h)):
    t = np.linspace(t_start, t_end, int((t_end-t_start)/h[i]))
    improved_euler_solution = improved_euler(f, ic, t)
    heun_solution = heun(f, ic, t)
    analytic_solution = analytic_solution(t)

    err_euler_improved = np.max(abs(analytic_solution-improved_euler_solution))
    err_heun = np.max(analytic_solution-heun_solution)

plt.figure(2)
plt.loglog(h,err_euler_improved, h, err_heun, h, h*h)
plt.show()

