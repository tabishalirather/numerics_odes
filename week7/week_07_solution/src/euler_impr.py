import numpy as np

# improved Euler method for multidimensiona ODEs
def euler_impr(f,t,y0):    
  y = np.zeros((len(t),len(y0)));        # initialize y 
  y[0,:] = y0                            # store the initial value y0 as the first row of y  
  for n in range(len(t)-1):  
    h = t[n+1]-t[n]                      # current step size
    y_half = y[n,:] + h/2*f(t[n],y[n,:])     # intermediate half step
    y[n+1] = y[n,:] + h*f(t[n]+h/2,y_half) # improved Euler method (for one-dimensional ODEs)

  return y