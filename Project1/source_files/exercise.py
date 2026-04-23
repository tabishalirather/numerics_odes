from collections.abc import Iterator

import numpy as np
# from numpy.random.c_distributions import random_standard_t
from scipy.integrate import odeint
from scipy.integrate import odeint, solve_ivp


# from scipy.stats import alpha


def ode_system(t, h_t, z_t, r_t, pa=0, pb=0,):
    """
    – α, dealing with human–zombie encounters that remove zombies.
    – β, dealing with human–zombie encounters that convert humans to zombies
    - dh_dt = -beta*h_t*z_t
    - dz_dt =  beta*h_t*z_t - alpha*h_t*z_t
    - dr_dt = alpha*h_t*z_t
    """

    beta = pb * 1
    alpha = pa * 1
    # h_t = np.linspace(0, 10, 10)
    # z_t = np.linspace(0, 10, 10)
    # r_t = np.linspace(0, 10, 10)
    dh_dt = -beta * h_t * z_t
    dz_dt = beta * h_t * z_t - alpha * h_t * z_t
    dr_dt = alpha*h_t*z_t

def zombie(strategy, pa, pb):
    """
    zombie(strategy,pa,pb)
    
    This function computes the evolution of the populations of humans and zombies 
    according to a chosen mitigation strategy.
    
    Parameters
    -----------------
    strategy : integer
        mitigation strategy
        0 : no intervention
        1 : training and arming humans
        2 : vaccination of humans 
    pa : real number
        uncertainty in the paramater alpha (= efficiency of humans killing zombies)
    pb : real number
        uncertainty in the paramater beta (= efficiency of zombies infecting humans)

    Returns
    -----------------
    t : numpy.array
        vector containing the time grid of the integration period of 1 year, t = [0,...,1]
    x : numpy.array 
        x[i,0] : number of humans at time t[i]      
        x[i,1] : number of zombies at time t[i]      
        x[i,2] : number of removed zombies at time t[i]      
    
    Example
    -----------------
    >>> t, x = zombie(0,0,0)        # Baseline scenario without intervention and with given values of alpha and beta  
    >>> t, x = zombie(1,-0.25,0.25) # Strategy 1 in worst case scenario where alpha is 25% decreased and beta is 25% increased.  
    """
    '''
    Defining Vocab: 
    1. h_t: humans 
    2. z_t: zombies
    3. r_t: removed/dead zombine 
    dh_dt = -beta*h_t*z_t
    dz_dt =  beta*h_t*z_t - alpha*h_t*z_t
    dr_dt = alpha*h_t*z_t
    
    x[i,0] : h_t[i]      
    x[i,1] : z_t[i]      
    x[i,2] : r_t[i]
    '''

    ####################
    # edit the code here#
    ####################
    t = np.linspace(0, 1, 100)

    if (0 == strategy):
        print("We did nothing")
    elif (1 == strategy):
        print("we used srategy 1, training and arming humans")
    elif (2 == strategy):
        print("we used strategy 2, vaccinations.")

    return strategy
