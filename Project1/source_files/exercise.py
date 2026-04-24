# from collections.abc import Iterator
from logging import warn, warning

import numpy as np
# from numpy.random.c_distributions import random_standard_t
from scipy.integrate import odeint
from scipy.integrate import odeint, solve_ivp


# from scipy.stats import alpha


def ode_system(y, t, alpha, beta):
    """
    – α, dealing with human–zombie encounters that remove zombies. Zombie removal quotient
    – β, dealing with human–zombie encounters that convert humans to zombies. Human infection quotient
    - dh_dt = -beta*h_t*z_t
    - dz_dt =  beta*h_t*z_t - alpha*h_t*z_t
    - dr_dt = alpha*h_t*z_t
    """

    # alpha = 0.5
    # alpha = alpha*(1 + pa)
    # # warning(f"Uncertainity in alpha is: {pa}")
    # # beta = 0.6
    # beta = beta*(1 + pb)
    # warning(f"Uncertainity in alpha is: {pb}")
    h_t, z_t, r_t = y
    # h_t = max(h_t, 0.0)
    # z_t = max(z_t, 0.0)
    # r_t = max(r_t, 0.0)
    dh_dt = -beta * h_t * z_t
    dz_dt = (beta * h_t * z_t) - (alpha * h_t * z_t)
    dr_dt = alpha * h_t * z_t
    return [dh_dt, dz_dt, dr_dt]


def zombie(strategy, pa, pb):
    """
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
    zombie(strategy,pa,pb)
    
    This function computes the evolution of the populations of humans and zombies 
    according to a chosen mitigation strategy.
    
    Parameters
    -----------------
    strategy : integer
        mitigation strategy
        0 : no intervention
        1 : training and arming humans -> ##increase alpha
        2 : vaccination of humans -> ##reduce beta
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
    ####################
    # edit the code here#
    ####################

    initial_humans = 1000
    initial_zombies = 10
    t = np.linspace(0, 1, 101)
    ALPHA = 0.5  # efficientcy of killling zombies, increased by trainign in arming by 50%
    # alpha_list = [alpha, alpha + (0.5 * alpha)]
    BETA = 0.6  # infection rate, reduced by half with vaccines.
    # beta_list = [beta, beta-(0.5*beta)] #reduce beta by half with strategy

    if (1 == strategy):
        alpha = ALPHA + (0.5 * ALPHA)
        beta = BETA
    elif (2 == strategy):
        alpha = ALPHA
        beta = BETA - (0.5 * BETA)
    else:
        alpha = ALPHA
        beta = BETA

    alpha = alpha * (1 + pa)
    beta = beta * (1 + pb)


    y_0 = [initial_humans, initial_zombies, 0]
    x = odeint(
        ode_system,
        y_0,
        t,
        args=(alpha, beta),
        rtol = 1e-9,
        atol=1e-12
    )
    # h_t = x[:, 0]
    # z_t = x[:, 1]
    # r_t = x[:, 2]

    return t, x
