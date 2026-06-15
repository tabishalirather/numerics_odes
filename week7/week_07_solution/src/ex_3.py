# script that runs improved Euler (with constant step size) and
# Adams-Bashforth 2-step method (AB-2)
# and compares accuracy versus runtime
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.integrate import solve_ivp
from twostep import twostep
from euler_impr import euler_impr

# Define a range step numbers for equidistant time grids.
N_v = [1e3, 1e4, 1e5, 1e6]

t0 = 0               # integration interval [t0,te] 
te = 50
y0 = np.array([0,2]) # initial condition

# Parameters for AB-2
a = [0, -1, 1]
b = [-0.5, 1.5, 0]

# various values for the factor mu appearing in the ODE function
# y'' -  mu*(1-y^2)y'+ y = 0.
mu = np.array([0.1,1,5])

for k in range(len(mu)):
  mu_k = mu[k]
  # define right-hand side of Van der Pol system y_0 := y, y_1:= y':
  #   y_0' = y_1  
  #   y_1' = y_0'' = mu_k * (1-y_0^2)y_1 - y_0
  vanderpol = lambda t,y: np.array([ y[1], mu_k*(1-y[0]**2)*y[1]-y[0] ]) 
  err = np.zeros((2,len(N_v)))    # initialize array of global errors
  times  = np.zeros((2,len(N_v))) # initialize array of runtimes
  y = solve_ivp(vanderpol,[t0,te],y0,dense_output=True,rtol = 1e-6) # approximate exact solution computed by solve_ivp
  ye = y.y.T[-1,:]                # approximate solution at te
    
  for n in range(len(N_v)):
    Nsteps = int(N_v[n])+1                        # number of time steps 
    t = np.linspace(t0,te,Nsteps)                 # corresponding equidistant grid
    start = time.time()                           # 
    z = solve_ivp(vanderpol,[t[0],t[1]],y0)       # compute the second starting value y1 with solve_ivp 
    y1 = z.y.T[-1,:]                              # y1 := last entry (last row) of the solution z.y 
                                                  #     = approximate solution at t[1]
    y_twostep = twostep(vanderpol,t,y0,y1,a,b)    # compute numerical solution with two-step method method
    end = time.time() 
    times[0,n] = end - start                      # runtime of two-step method
    err[0,n] = np.max(np.abs(ye-y_twostep[-1,:])) # global error at te
    start = time.time()
    y_impr = euler_impr(vanderpol,t,y0)           # compute numerical solution with improved Euler method
    end = time.time() 
    times[1,n] = end - start                      # runtime of improved Euler with constant step size
    err[1,n] = np.max(np.abs(ye-y_impr[-1,:]))    # global error at te    
    
  # function plot of y_1(t)
  fs = 15
  plt.rcParams.update({'font.size': fs})
  plt.figure()
  z = y.sol(t)
  z = z.T
  plt.plot(t,z[:,0],'k',t, y_twostep[:,0],'-.b',t, y_impr[:,0],'--g')
  plt.legend(['RK45','AB-2','improved Euler'], loc = 'best')
  plt.title(r'function plot for $\mu =$' + str(mu_k))
  plt.xlabel('t')
  plt.ylabel('y')    

  # plot global error vs runtime
  plt.figure()
  plt.loglog(times[0,:],err[0,:],'-ob',times[1,:],err[1,:],'-og')
  plt.xlabel('error')
  plt.ylabel('time [s]')
  plt.legend(['AB-2','improved Euler','Location'],loc = 'best')   
  plt.title(r'runtime vs global error for $\mu =$' + str(mu_k))
