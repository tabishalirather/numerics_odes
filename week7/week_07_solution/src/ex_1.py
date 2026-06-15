import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from twostep import twostep

# IVP: y'(t) = -y, t in [0,5]   
#      y(0)  = 1      
f = lambda t,y:  -y
y0 = np.array([1]) 
t0 = 0
te = 5

y_exact = lambda t: y0*np.exp(-t)

# # another IVP: y'(t) = 1/(1+t^2), t in [0,4]   
# #              y(0)  = 0      
# f = lambda t,y: 1/(1 + t**2)
# y0 = np.array([0])  
# t0 = 0
# te = 4

# y_exact = lambda t: np.arctan(t)

# Parameters for Adams-Bashforth two-step method (AB-2)
a = [0, -1, 1];
b = [-0.5, 1.5, 0];

# Define array of step sizes:
h = np.logspace(-3,-1,20)
n = len(h)
err = np.zeros(n) # initialize array for global errors

# Compute for each step size h[i] a numerical solution with the two step method
# and its corresponding global error.
for i in range(n):
  N = int((te-t0)/h[i])                  # rounded number of steps needed for step size h[i]
  t = np.linspace(t0,te,N+1)             # equidistant time grid with (roughly) stepsize h[i]
  t = np.reshape(t,(len(t),1))           # reshape t to a column vector
  z = solve_ivp(f,[t[0],t[1]],y0)        # compute the second starting value y1 with solve_ivp 
  y1 = z.y.T[-1,:]                       # y1 := last entry (last row) of the solution z.y 
                                         #     = approximate solution at t[1]
  y_twostep = twostep(f,t,y0,y1,a,b)     # numerical result computed by two-step method with stepsize h[i]
  y = y_exact(t)                         # exact solution  
  err[i] = np.max(np.abs(y_twostep - y)) # global error for stepsize h[i]

fs = 15
plt.rcParams.update({'font.size': fs})
plt.figure()
plt.loglog(h,err,'b+',h,h**2,'--r')  
plt.legend(['global error','straight line of slope 2'],loc = 'best')
plt.xlabel('step size, h')
plt.ylabel('error')
plt.title('error vs. stepsize')
plt.grid()

# verify convergence order by fitting a line through the data
pp = np.polyfit(np.log(h), np.log(err), 1)
print('measured slope: ' + str(round(pp[0],2)))