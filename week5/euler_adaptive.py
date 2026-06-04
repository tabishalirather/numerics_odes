# Euler method with step size control 
#
# y' = y^2, y(0.8) = 5/6 for x in [0.8,1.8].
# solution: y(x) = 1/(2-x).
#
# Author: Christian Seifert
# Modifications: Daniel Ruprecht
# Modifications: Florian Bünger

import numpy as np
import matplotlib.pyplot as plt

f = lambda x,y: y**2 # right-hand side of ODE y' = y^2 
t0 = 0.8             # start time
y0 = 5/6             # initial value y(t0) = y0
te = 1.8             # end time
tol = 1.e-3          # tolerance for step size control
n = 0                # step counter
t = np.array([t0])   # initial condition 
y = np.array([y0])   # y(t0) = y0
hstart = 0.1         # guess for a start stepsize
h = hstart 
htmp = h
while htmp > 0:
  y1 = y[n] + htmp*f(t[n],y[n])
  y2 = y[n] + htmp/2*f(t[n],y[n])
  y2 = y2 + htmp/2 * f(t[n]+htmp/2,y2)
  phi = 2 * np.linalg.norm(y2-y1)
  # Note how the change in step size is limited to a minimum factor of 0.2 
  # and a maximum factor of 10.
  hneu = htmp * np.min([np.max([0.9*np.sqrt(tol/phi),0.2]),10])
  if phi > tol:
    htmp = hneu
  else:
    h = htmp
    t = np.append(t,t[n]+h)        
    y = np.append(y,2*y2-y1)        # Use the value provided by the more accurate scheme (which equals the improved Euler method). 
    htmp = np.min([te-t[n+1],hneu]) # Reduce the time stepsize if close to end time te
    n = n+1

# Plot numeric and exact solution 
fs = 15 # font size for plots
y_exact = 1/(2-t)
plt.figure()                                                          
plt.plot(t,y,'bo',t,y_exact,'r') 
plt.rcParams.update({'font.size': fs})
plt.xlabel('t')
plt.ylabel('y')
plt.legend(['Euler adapative step control','exact solution'],loc = 'best')

print('Euler adapative step control -- Number of steps: ',n,'  Maximum error: ',"{:.4e}".format(np.max(np.abs(y-y_exact))))






