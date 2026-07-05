import numpy as np
from scipy.integrate import odeint
from scipy.linalg import expm
import matplotlib.pyplot as plt

def runge_kutta(f,Jf,y0,t,a,B,c):
  """
  runge_kutta(f,y0,t,a,B,c)    
  
  This function implements explicit or diagonally implicit Runge-Kutta methods for solving the initial value problem
  
    y'(z) = f(z,y(z))  for z in [t0,te]        
    y(t0) = y0  
    
  where t0 is the first and te is the last entry in the array t.
    
  Parameters
  -----------------
  f    : callable(t,y)
         Computes the derivative of y at t, i.e., computes the right-hand side of y' = f(t,y). 
  Jf   : callable(t,y) for implicit method, and [] (empty) for explicit method
         For an implicit method JF is the Jacobian of f with respect to y,
         i.e., Jf(t,y) = df/dy(t,y).         
  y0   : numpy.array
         Initial condition on y (can be a vector).
  t    : numpy.array
         A sequence of time points for which to solve for y. 
         The initial value point must be the first element of this sequence. 
         This sequence must be monotonically increasing.
  a,B,c: coefficients of the Butcher tableau of the Runge-Kutta method  
         which sall be executed:
      
           a | B          
           ------
             | c
           
         Here, a and c are numpy vectors of length s, where s is the number
         of stages, and B is an s-by-s numpy array
       
  Returns
  -----------------
  y : numpy.array, shape (len(t), len(y0))
      Array containing the value of y for each desired time in t, 
      with the initial value y0 in the first row.
    
  Example
  -----------------
  >>> f = lambda t,y: y/2
  >>> Jf = lambda t,y: 1/2            # df/dy = 1/2
  >>> y0 = np.array([1])
  >>> t = np.linspace(0,3,100)
  >>> # explicit method
  >>> a = np.array([0,1])             # Heun's method
  >>> B = np.array([[0,0],[1,0]])     # Heun's method
  >>> c = np.array([1/2,1/2])         # Heun's method
  >>> y = runge_kutta(f,[],y0,t,a,B,c)    
  >>> # diagonally implicit method
  >>> a = np.array([0,1])             # Trapezoidal rule
  >>> B = np.array([[0,0],[1/2,1/2]]) # Trapezoidal rule
  >>> c = np.array([1/2,1/2])         # Trapezoidal rule
  >>> y = runge_kutta(f,Jf,y0,t,a,B,c)  
  """  
  
  ####################
  #edit the code here#
  ####################
    
  return ##

def k_step(f,Jf,y0,y1,t,a,b):
  """
  k_step(f,y0,y1,t,a,b)    
  
  This function implements an explicit or implicit k-step method for solving the initial value problem
  
    y'(s) = f(s,y(s))  for s in [t0,te]        
    y(t0) = y0  
    
  where t0 is the first and te is the last entry in the array t.
    
  Parameters
  -----------------
  f  : callable(t,y)
       Computes the derivative of y at t, i.e., computes the right-hand side of y' = f(t,y). 
  Jf : callable(t,y) for implicit method, and [] (empty) for explicit method
       For an implicit method JF is the Jacobian of f with respect to y,
       i.e., Jf(t,y) = df/dy(t,y).
  y0 : numpy.array
       Initial condition on y (can be a vector).
  y1 : numpy.array, shape (k-1,len(y0))
       starting values for time grid points t[1],...,t[k-1].      
       This is an optional parameter, i.e., if y1 is empty,   
       then the starting values are computed (by odeint, for example).
  t  : numpy.array
       A sequence of time points for which to solve for y. 
       The initial value point must be the first element of this sequence. 
       This sequence must be monotonically increasing. 
  a,b: Parameters of the k-step method
       a_0 * y_n + ... + a_k * y_k = h * ( b0 * f(t_n,y_n) + ... + b_k * f(t_{n+k},y_{n+k}) )
      
  Returns
  -----------------
  y : numpy.array, shape (len(t),len(y0))
      Array containing the value of y for each desired time in t, 
      with the initial value y0 in the first row.
    
  Example
  -----------------
  >>> f  = lambda t,y: y/2                # y' = f(t,y,c) = y/2
  >>> Jf = lambda t,y: 1/2                # df/dy = 1/2
  >>> y0 = np.array([1])                  # y0 = y(0) = 1 ---> y(t) = exp(t/2)                   
  >>> t = np.linspace(0,3,100)   
  >>> # explicit two-step method 
  >>> a = np.array([0, -1, 1])            # Adams-Bashforth-2 two-step method
  >>> b = np.array([-0.5, 1.5, 0])        # Adams-Bashforth-2 two-step method       
  >>> y = k_step(f,[],y0,[],t,a,b)    
  >>> # implicit three-step method      
  >>> a = np.array([-1/3, 3/2, -3, 11/6]) # BDF-3
  >>> b = np.array([0, 0, 0, 1])          # BDF-3    
  >>> y = k_step(f,Jf,y0,[],t,a,b)   
  """ 
  
  ####################
  #edit the code here#
  ####################
    
  return ##

def f(t,w,R,S):
  """
  f(t,w,R,S)    
  
  This function implements the right-hand side of the sought ODE w' = f(t,w,R,S) 
  for the particle motion in Project 3, see the project 
  description sheet in StudIP for details.    
    
  Parameters
  -----------------
  t : numpy.array, real number 
      time point(s)        
  w : numpy.array 
      w = [y,v]
      y : particle position at time t
      v : particle velocity at time t          
  R : positive real number
      R < 1: particles lighter than fluid
      R = 1: neutrally buoyant
      R > 1: particles denser than fluid 
  S : positive real number, measure for particle size     
    
  Returns
  -----------------
  dwdt : numpy.array
         dwdt = f(t,w) = [dydt,dvdt]      
  """ 
  
  ####################
  #edit the code here#
  ####################
  y1, y2, v1, v2 = w
  P = 1.0 / (R * S)

  dy1dt = v1
  dy2dt = v2
  dv1dt = -P * v1 - v2 / R - y2 / (R * S)
  dv2dt = y1 / (R * S) + v1 / R - P * v2

  dwdt = np.array([dy1dt, dy2dt, dv1dt, dv2dt])

  return dwdt

def Jf(t,w,R,S):
  """
  Jf(t,w,R,S)    
  
  This function implements the Jacobian of the previously defined function f(t,w,R,S),
  i.e., Jf(t,w,R,S) is the derivative of f(t,w,R,S) with respect to w. 
    
  Parameters
  -----------------
  t : real number  
      time point        
  w : numpy.array 
      w = [y,v]
      y : particle position
      v : particle velocity
  R : positive real number
      R < 1: particles lighter than fluid
      R = 1: neutrally buoyant
      R > 1: particles denser than fluid 
  S : positive real number, measure for particle size             
      
  Returns
  -----------------
  dfdc : numpy.array shape (4,4) , i.e., 4-by-4 matrix 
         dfdw = (df[i]_dw[j])_{0 <= i,j <= 3} 
  """  
  
  ####################
  #edit the code here#
  ####################
  P = 1.0 / (R * S)

  dfdw = np.array([
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0, -1 / (R * S), -P, -1 / R],
    [1 / (R * S), 0, 1 / R, -P],
  ])

  return dfdw

def w_exact(t,R,S,w0):
  print("calling w_exact")
  """
  w_exact(t,R,S,w0)    
  
  This function implements the analytical solution of the IVP 
    w' = f(t,w,R,S), w(0) = w0. 
    
  Parameters
  -----------------
  t : numpy.array
      A sequence of time points for which to solve for w. 
      The initial value point must be the first element of this sequence. 
      This sequence must be monotonically increasing.      
  R : positive real number
      R < 1: particles lighter than fluid
      R = 1: neutrally buoyant
      R > 1: particles denser than fluid 
  S : positive real number, measure for particle size             
      
  Returns
  -----------------
  w : numpy.array, shape (len(t), len(w0))
      Array containing the value of w for each desired time in t, 
      with the initial value w0 in the first row.
  """     
    
  ####################
  #edit the code here#
  ####################
  t = np.asarray(t, dtype=float)
  y10, y20, v10, v20 = w0

  a = 1.0 / R
  b = 1.0 / (R * S)

  disc = (b - 1j * a) ** 2 + 4j * b
  lam1 = ((1j * a - b) + np.sqrt(disc)) / 2
  lam2 = ((1j * a - b) - np.sqrt(disc)) / 2

  Z0 = y10 + 1j * y20
  V0 = v10 + 1j * v20

  D1 = (V0 - lam2 * Z0) / (lam1 - lam2)
  D2 = (lam1 * Z0 - V0) / (lam1 - lam2)

  Zt = D1 * np.exp(lam1 * t) + D2 * np.exp(lam2 * t)
  Vt = lam1 * D1 * np.exp(lam1 * t) + lam2 * D2 * np.exp(lam2 * t)

  y1 = np.real(Zt)
  y2 = np.imag(Zt)
  v1 = np.real(Vt)
  v2 = np.imag(Vt)

  w = np.column_stack([y1, y2, v1, v2])
  return w


def phase_plot(t):
  w79 = w_exact(t, 7 / 9, 0.3, w0)  # R < 1, particle lighter than fluid
  w43 = w_exact(t, 4 / 3, 0.3, w0)  # R > 1, particle heavier than fluid

  fig, axs = plt.subplots(1, 2, figsize=(8, 4))
  axs[0].plot(w79[:, 0], w79[:, 1])
  axs[0].set_xlabel('$y_1$');
  axs[0].set_ylabel('$y_2$')
  axs[0].set_title('R = 7/9, S = 0.3')

  axs[1].plot(w43[:, 0], w43[:, 1], 'g')
  axs[1].set_xlabel('$y_1$');
  axs[1].set_ylabel('$y_2$')
  axs[1].set_title('R = 4/3, S = 0.3')