import numpy as np

def twostep(f,t,y0,y1,a,b):
# general two-step method.
# Inputs required:
#
# f = anonymous function for the right-hand side of the ODE y' = f(t,y)
# t = equdistant time grid
# y0 = initial condition (y-coordinate)
# y1 = approximate for y(t+h)
# a  = coefficients for general twostep method
# b  = coefficients for general twostep method

  dim = len(y0)  # dimension of the ODE-system  
  h = t[1]-t[0]  # stepsize of equdistant grid t
  y = np.zeros((len(t),dim)) # initialize solution

  # Plug initial conditions:
  y[0,:] = y0
  y[1,:] = y1

  # Execute two step method: 
  for n in range(len(t)-2):
    y[n+2,:] = -a[1]*y[n+1,:] - a[0]*y[n,:] + h * ( b[0]*f(t[n],y[n,:]) + b[1] * f(t[n+1],y[n+1,:]) ) 

  return y