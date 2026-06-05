import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# -------------------------------------------------------------------------------------------
# Constants (do not change!)
# -------------------------------------------------------------------------------------------
V = 25  # biochemical reactor volume [m^3]
Q1 = 2.5  # discharge of the first pipe [m^3/min]
Q2 = 2.5  # discharge of the second pipe [m^3/min]
cz0 = 20  # virus concentration in the first pipe [mg/m^3]
cc0 = 25  # chemical concentration in the second pipe [mg/m^3]
k = 1.2  # reaction rate constant [m^3/(mg*min)]
Q3 = Q1 + Q2


# -------------------------------------------------------------------------------------------
# (1) Template for the explicit Euler method that you shall complete.
# -------------------------------------------------------------------------------------------
def euler_expl(f, y0, t):
    """
    euler_expl(f,y0,t)

    This function implements the explicit Euler method for solving the initial value problem

      y'(s) = f(s,y(s))  for s in [t0,te]
      y(t0) = y0

    where t0 is the first and te is the last entry in the array t.

    Parameters
    -----------------
    f  : callable(t,y)
         Computes the derivative of y at t, i.e., computes the right-hand side of y' = f(t,y).
    y0 : numpy.array
         Initial condition on y (can be a vector).
    t  : numpy.array
         A sequence of time points for which to solve for y.
         The initial value point must be the first element of this sequence.
         This sequence must be monotonically increasing.

    Returns
    -----------------
    y : numpy.array, shape (len(t), len(y0))
        Array containing the value of y for each desired time in t,
        with the initial value y0 in the first row.

    Example
    -----------------
    >>> f = lambda t,y: y/2
    >>> y0 = np.array([1])
    >>> t = np.linspace(0,3,100)
    >>> y = euler_expl(f,y0,t)
    """

    ####################
    # edit the code here#
    ####################
    dim = len(y0)
    y = np.zeros((len(t), dim))
    y[0] = y0
    for n in range(len(t) - 1):
        h = t[n + 1] - t[n]
        y[n + 1] = y[n] + (h * f(t[n] + h, y[n]))
    return y


# -------------------------------------------------------------------------------------------
# (2) Template for the improved Euler method that you shall complete.
# -------------------------------------------------------------------------------------------
def euler_impr(f, y0, t):
    """
    euler_impr(f,y0,t)

    This function implements the explicit Euler method for solving the initial value problem

      y'(s) = f(s,y(s))  for s in [t0,te]
      y(t0) = y0

    where t0 is the first and te is the last entry in the array t.

    Parameters
    -----------------
    f  : callable(t,y)
         Computes the derivative of y at t, i.e., computes the right-hand side of y' = f(t,y).
    y0 : numpy.array
         Initial condition on y (can be a vector).
    t  : numpy.array
         A sequence of time points for which to solve for y.
         The initial value point must be the first element of this sequence.
         This sequence must be monotonically increasing.

    Returns
    -----------------
    y : numpy.array, shape (len(t), len(y0))
        Array containing the value of y for each desired time in t,
        with the initial value y0 in the first row.

    Example
    -----------------
    >>> f = lambda t,y: y/2
    >>> y0 = np.array([1])
    >>> t = np.linspace(0,3,100)
    >>> y = euler_impr(f,y0,t)
    """

    ####################
    # edit the code here#
    ####################
    dim = len(y0)
    y = np.zeros((len(t), dim))
    y[0] = y0

    for n in range(len(t) - 1):
        h = t[n + 1] - t[n]
        h_half = h / 2
        y_half = y[n] + (h_half) * f(t[n], y[n])
        y[n + 1] = y[n] + h * f(t[n] + h_half, y_half)
    return y


# -------------------------------------------------------------------------------------------
# (3) Template for a general explicit or implicit two-step method that you shall complete.
# -------------------------------------------------------------------------------------------
def twostep(f, Jf, y0, y1, t, a, b):
    """
    twostep(f,y0,y1,t,a,b)

    This function implements an explicit or implicit two-step method for solving the initial value problem

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
    y1 : numpy.array
         approximate solution for the second time grid point t[1].
         This is an optional parameter, i.e., if y1 is empty,
         then an approximate value for y1 is computed.
    t  : numpy.array
         A sequence of time points for which to solve for y.
         The initial value point must be the first element of this sequence.
         This sequence must be monotonically increasing.
    a,b: Parameters of the two-step method
           a_2*y_{n+2} + a_1*y_{n+1} + a_0*y_n =
           h * (b_0*f(t_n,y_n) + b_1*f(t_{n+1},y_{n+1}) + b_2*f(t_{n+2},y_{n+2}))
         In case of an explicit method a_2 = 1 and b_2 = 0 (or len(b)=2) is assumed.

    Returns
    -----------------
    y : numpy.array, shape (len(t),len(y0))
        Array containing the value of y for each desired time in t,
        with the initial value y0 in the first row.

    Example
    -----------------
    >>> f  = lambda t,y: y/2             # y' = f(t,y,c) = y/2
    >>> Jf = lambda t,y: 1/2             # df/dy = 1/2
    >>> y0 = np.array([1])               # y0 = y(0) = 1 ---> y(t) = exp(t/2)
    >>> t = np.linspace(0,3,100)
    >>> # explicit two-step method
    >>> a = np.array([0, -1, 1])         # Adams-Bashforth-2 two-step method
    >>> b = np.array([-0.5, 1.5, 0])     # Adams-Bashforth-2 two-step method
    >>> y = twostep(f,[],y0,[],t,a,b)
    >>> # implicit two-step method
    >>> a = np.array([0, -1, 1])         # Adams-Moulton-2 two-step method
    >>> b = np.array([-1/12, 2/3, 5/12]) # Adams-Moulton-2 two-step method
    >>> y = twostep(f,Jf,y0,[],t,a,b)
    """

    ####################
    # edit the code here#
    ####################
    dim = len(y0)
    y = np.zeros((len(t), dim))
    # print(y0)
    y[0] = y0
    # if y1 is False:
    if y1 == []:
        # print(f"y1 is mepty")
        y[1] = odeint(f, y0, t, tfirst=True)[1]
        # We can create y1 by solve IVP here

    if len(b) == 2 or b[2] == 0:
        pass  # Explicity two step method
    else:
        for n in range(len(t) - 2):
            h = t[n + 1] - t[n]
            # print(f"b[0] is: {type(b[0])}")
            # print(f"(b[0] * f(t[n], y[n]): {f(t[n], y[n])}")
            # b[0] = b[0] * np.identity(dim)
            # b[1] = b[1] * np.identity(dim)
            f_n = f(t[n], y[n])
            f_n1 = f(t[n + 1], y[n + 1])
            # Here why am I not able to do b0 * fn? this is simple scalar multiplicaiton of avactor
            const_S = a[0] * y[n] + a[1] * y[n + 1] - h * (b[0] * f_n + b[1] * f_n1)

            def F(y):
                return a[2] * y - h * b[2] * f([t[n + 2]], y) + const_S

            def JF(y):
                return a[2] * np.identity(dim) - h * b[2] * Jf(t[n + 2], y)

            z_initial_guess = y[n + 1]
            newton_update = 1 * np.identity(dim)  # Das ist eine vector
            while np.linalg.norm(newton_update) < 1e-8:
                newton_update = np.linalg.solve(JF(z_initial_guess), F(z_initial_guess))  #
                z_initial_guess = z_initial_guess - newton_update
            y[n + 2] = z_initial_guess
            # y = np.array(y)
    return y


# -------------------------------------------------------------------------------------------
# (4) Template for the ODE function that you shall complete.
# -------------------------------------------------------------------------------------------
def f(t, c):
    """
    f(t,c)

    This function implements the right-hand side of the sought ODE c' = f(t,c)
    for the biochemical reaction considered in Project 2, see the project
    description sheet in StudIP for details.

    Parameters
    -----------------
    t : numpy.array, real number
        time point(s)
    c : numpy.array
        c = [cz,cc,cp]
        cz : zombie virus concentration
        cc : chemical concentration
        cp : vaccine product concentration

    Returns
    -----------------
    dcdt : numpy.array
           dcdt = f(t,c) = [dcz_dt,dcc_dt,dcp_dt]
    """
    ####################
    # edit the code here#
    ####################
    cz, cc, cp = c
    dcz_dt = (Q1 * cz0) - (Q3 * cz) - (k * V * cc * cz)
    # print(f"dcz_dt : {dcz_dt}")

    dcc_dt = (Q2 * cc0) - (Q3 * cc) - (k * V * cc * cz)
    # print(f"dcc_dt: {dcc_dt}")

    dcp_dt = 0 - (Q3 * cp) + 2 * k * cc * cz
    # print(f"dcc_dt: {dcc_dt}")
    dcdt = [dcz_dt, dcc_dt, dcp_dt]
    dcdt = np.array(dcdt)

    return dcdt


# -------------------------------------------------------------------------------------------
# (5) Template for the Jacobian of the ODE function that you shall complete.
# -------------------------------------------------------------------------------------------
def Jf(t, c):
    """
    Jf(t,c)

    This function implements the Jacobian of the previously defined function f(t,c),
    i.e., Jf(t,c) is the derivative of f(t,c) with respect to c.

    Parameters
    -----------------
    t : real number
        time point
    c : numpy.array
        c = [cz,cc,cp]
        cz : zombie virus concentration
        cc : chemical concentration
        cp : vaccine product concentration

    Returns
    -----------------
    dfdc : numpy.array shape (3,3) , i.e., 3-by-3 matrix
           dfdc = [df[0]_dc[0], df[0]_dc[1], df[0]_dc[2],
                   df[1]_dc[0], df[1]_dc[1], df[1]_dc[2],
                   df[2]_dc[0], df[2]_dc[1], df[2]_dc[2]]
    """

    ####################
    # edit the code here#
    ####################
    cz, cc, cp = c
    dfdc = [
        -Q3/V - k*cc, -k*cz , 0,
        -k*cc, -Q3/V - k*cz, 0,
        2*k*cc, 2*k*cz, -Q3
    ]

    dfdc = np.array(dfdc)
    return  dfdc
