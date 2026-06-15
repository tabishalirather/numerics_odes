import numpy as np
import matplotlib.pyplot as plt
from twostep import twostep

f = lambda t,y: 0          # IVP y'(t) = 0, y(0) = 1
t0 = 0                     # integration interval
te = 3                     # some end of integration (not given)
y0 = np.array([1])         # initial condition
y1 = np.array([1 + 1e-15]) # first time step, use 1e-13 to prevent B from being close to machine precision
    
N = 22                     # number of steps
t = np.linspace(t0,te,N+1) # time grid
h = t[1]-t[0]              # step size

a = [-5, 4, 1] # parameters for two-step method 
b = [2, 4, 0]
y_twostep = twostep(f,t,y0,y1,a,b) # execute two step method

# Determine constants A and B satisfying  
#    y_n = A + B*(-5)^n
#    y_0 = 0 
#    y_1 = 1 + 1e-15
#
# This leads to the 2-by-2 linear system 
# 
#   A + B  = 1  
#   A - 5B = 1 + 1e-15
# 
#  which is written as Cx = Y.

C = np.array([[1,1],[1,-5]])
Y = np.array([y0,y1])
x = np.linalg.solve(C,Y)
# x = C\Y 
A = x[0]
B = x[1]

# # direct setting of A and B:
# A = (5*y0+y1)/6 
# B = (y0-y1)/6 

# check if correct
NN = np.arange(N+1)
y = A+B*(-5)**NN

# plot to verify
fs = 15
plt.rcParams.update({'font.size': fs})
plt.figure()
plt.plot(t,y_twostep,'b+',t,y,'rx',t,np.ones(len(t)),'k')
plt.xlabel('t')
plt.ylabel('y')

plt.legend(['two step method','direct calculation','exact solution = const. 1'],loc = 'best')



