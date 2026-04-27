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

def f(t, y, _d=d):
    return _d*y - y**2

euler_explicit_value = euler_explicit(initial_condition, t, f)
y_analytic_value = y_analytic(t)
print(euler_explicit_value)
print(y_analytic_value)


plt.figure(1)
plt.plot(t, euler_explicit_value, 'r', t, y_analytic_value, '--b')
plt.legend(['Explicit Euler', 'Analytic Soln'])
plt.show()


# Global ERROR
# for n in range(1,len(t)):
global_error = y_analytic_value - euler_explicit_value
plt.figure(2)
plt.plot(t, global_error, '--')
plt.show()

max_global_error = np.max(np.abs(global_error))
print(f"max_global_error is: {max_global_error}")

# Lipschitz constant:
L = d+2*(np.max(np.abs(y_analytic_value)))
K = np.max(np.abs((d-2*y_analytic_value*y_analytic_value)*(d*y_analytic_value)-y_analytic_value**2))
C = np.exp(L*(end_time - initial_time)) * K/2 * (end_time-initial_time)
h_step_size = t[1]-t[0]
print(f"K: {K}")
print(f"L: {L}")
print(f"L: {C}")

# Now plot this:
plt.figure(2)
plt.semilogy(t, np.abs(global_error), 'r', t, C*h_step_size+0*t, '--b')
plt.show()

num_step = [100, 1000, 1000]
errors_array = np.zeros(len(num_step))
step_sizes_h = np.zeros(len(num_step))

for i in range(len(num_step)):
    t = np.linspace(initial_time, end_time, num_step[i])
    y_euler = euler_explicit(initial_condition, t, f)
    y_analytic_value = y_analytic(t)
    errors_array[i] = np.max(np.abs(y_analytic_value - y_euler))
    step_sizes_h[i] = t[1] - t[0]

plt.figure(3)
plt.loglog(h_step_size, errors_array)
plt.show()