import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from exercise import *
import os

# ---------------------------------------------------------------------
# main program
# ---------------------------------------------------------------------


V = 25  # biochemical reactor volume [m^3]
Q1 = 2.5  # discharge of the first pipe [m^3/min]
Q2 = 2.5  # discharge of the second pipe [m^3/min]
cz0 = 20  # zombie virus concentration in the first pipe [mg/m^3]
cc0 = 25  # chemical concentration in the second pipe [mg/m^3]

k = 1.2  # reaction rate constant [m^3/(mg*min)]
t0 = 0
te = 30  # integration interval = 30 minutes
y0 = np.array([0, 0, 0])  # initial condition y0 = [0,0,0]

# Parameters for Adams-Bashforth-2 two-step method
a_AB2 = np.array([0, -1, 1])
b_AB2 = np.array([-0.5, 1.5, 0])

# Parameters for Adams-Moulton-2 two-step method
a_AM2 = np.array([0, -1, 1])
b_AM2 = np.array([-1 / 12, 2 / 3, 5 / 12])

# Parameters for BDF-2 two-step method 
a_BDF2 = np.array([1 / 2, -2, 3 / 2])
b_BDF2 = np.array([0, 0, 1])

num_solvers = 5
# h = np.array([0.1, 0.01, 1e-3, 1e-4])
h = np.array([0.1, 0.01])
cp_err = np.zeros((len(h), num_solvers))
y1_app_values = []
for i in range(len(h)):
    h_ = h[i]
    t = np.arange(t0, te, h_)
    t = np.append(t, te)
    y_app = odeint(f, y0, t, tfirst=True)
    # y1_app_values.append(y_app[1])

    # y1 = y_app[1]
    # print(f"y_app: {y_app[:5]}")
    # y1 = y_app[1]
    # y1 = np.array(y1)
    # print(f"this is what is being sent to twostep method as y1 param: {type(y1)}")
    y_euler_expl = euler_expl(f, y0, t)
    if y_euler_expl is not None:
        # print(f"y_euler_expl: {y_euler_expl[:5]}")
        pass
    else:
        print("y_euler_expl is empty")

    y_euler_impr = euler_impr(f, y0, t)
    if y_euler_impr is not None:
        # print(f"y_euler_impr: {y_euler_impr[:5]}")
        pass
    else:
        print("y_euler_impr is empty")
    y_AB2 = twostep(f, [], y0, [], t, a_AB2, b_AB2)
    if y_AB2 is not None:
        # print(f"f{y_AB2}: {y_AB2[:5]}")
        pass
    else:
        print("y_ABE is empty")

    y_AM2 = twostep(f, Jf, y0, [], t, a_AM2, b_AM2)
    if y_AM2 is not None:
        # print(f"y_AME: {y_AM2[:5]}")
        pass
    else:
        print("y_AME is emtpy")

    y_BDF2 = twostep(f, Jf, y0, [], t, a_BDF2, b_BDF2)
    if y_BDF2 is not None:
        pass
        # return print(f"y_BDFE: {y_BDF2[:5]}")
    else:
        print("y_BDF2 is empty")

    col = 2
    # print(f"y_euler_expl[:, col] {(y_euler_expl[:10, col])}")
    # print(f"y_app[:, col]): ({y_app[10, col]})")


    # absolute error for vaccine product concentration
    cp_err[i, :] = np.array([np.max(np.abs(y_euler_expl[:, col] - y_app[:, col])),
                             np.max(np.abs(y_euler_impr[:, col] - y_app[:, col])),
                             np.max(np.abs(y_AB2[:, col] - y_app[:, col])),
                             np.max(np.abs(y_AM2[:, col] - y_app[:, col])),
                             np.max(np.abs(y_BDF2[:, col] - y_app[:, col]))])


    # print(f"y1 using odeint is: {y_app[1]}")
    # print(f"y1 using y_AB2 is: {y_AB2[1]}")

# Display y(te) for final step size
print(f"cp_err: {cp_err}")
m = len(t) - 1
print('Concentrations c = (c_Z,c_C,c_P) at t_end = 30 min for h = ', h_, '\n')
print('     odeint : ', y_app[m, :])
print('expl. Euler : ', y_euler_expl[m, :])
print('impl. Euler : ', y_euler_impr[m, :])
print('        AB2 : ', y_AB2[m, :])
print('        AM2 : ', y_AM2[m, :])
print('       BDF2 : ', y_BDF2[m, :])

# --------------------------------------------------------------------------
# plot concentrations c_z(t), c_C(t), c_P(t)  [for h(end)]
# --------------------------------------------------------------------------

path = "/mnt/36C22184C2214987/Coursework/TUHH/Numerics for ODEs/Code/Project2/results/"
print(path)
# path = f"{cwd}/results" # <--- change this path to that of your folder
# if(os.path.exists(path) is False):
#   print("new path created")
#   path = os.path.join(path, 'results')
# else:
#   print("old path used")
#   path = path
#TODO: Fix AB2, it doesn't work; also BFD2 doesn't work, and AM2 doesn't work. Some mate, something is wrong with two step method;
fs = 15  # FontSize
title = ['Virus concentration', 'Chemical concentration', 'Vaccine product concentration']
ylabel = ['$c_Z$', '$c_C$', '$c_P$']
fname = ['c_Z.png', 'c_C.png', 'c_P.png']
for i in range(3):
    plt.figure()
    plt.rcParams.update({'font.size': fs})
    plt.plot(t,y_euler_expl[:,i],t,y_euler_impr[:,i],t,y_AB2[:,i],t,y_AM2[:,i],t,y_BDF2[:,i],t,y_app[:,i],'--k')
    # plt.plot(t, y_AM2[:, i], t, y_app[:, i], '--k')
    plt.legend(['Explicit Euler', 'Improved Euler', 'AB2', 'AM2', 'BDF2', 'odeint'], loc='best')
    # plt.legend(['other method', 'odeint'], loc='best')
    plt.title(title[i])
    plt.xlabel('t')
    plt.ylabel(ylabel[i])
    plt.savefig(path + fname[i])

# --------------------------------------------------------------------------
# plot global error c_P(t)
# --------------------------------------------------------------------------
plt.figure()
plt.rcParams.update({'font.size': fs})
plt.loglog(h, cp_err[:, 0], 'b', h, cp_err[:, 1], 'r', h, cp_err[:, 2], 'g', h, cp_err[:, 3], 'k', h, cp_err[:, 4], 'm')
plt.title('Error vs stepsize for c_P')
plt.legend(['Explicit Euler', 'Improved Euler', 'AB2', 'AM2', 'BDF2'], loc='best')
plt.xlabel('h')
plt.ylabel('Error')
plt.grid()
plt.savefig(path + 'error.png')

# --------------------------------------------------------------------------
# robustness
# --------------------------------------------------------------------------
h_ = 0.2
t = np.arange(t0, te, h_)
t = np.append(t, te)

y_euler_expl = euler_expl(f, y0, t)
y_euler_impr = euler_impr(f, y0, t)
y_AB2 = twostep(f, [], y0, [], t, a_AB2, b_AB2)
y_AM2 = twostep(f, Jf, y0, [], t, a_AM2, b_AM2)
y_BDF2 = twostep(f, Jf, y0, [], t, a_BDF2, b_BDF2)

# alpha = np.arange(0.5,1.5,0.05)
alpha = np.arange(0.7, 1.3, 0.05)
delta = np.zeros((num_solvers, len(alpha)))
j = 0  # c_z
# j = 1 # c_c
# j = 2 # c_P

k_ = np.array([-1, -1, 2]) * k

for i in range(len(alpha)):
    Q_in = np.array([alpha[i] * Q1, Q2, 0])  # variation of Q_in(1) = Q1 by at most +/- 30%
    Q_out = Q_in[0] + Q_in[1]
    c_in = np.array([cz0, cc0, 0])
    k_ = np.array([-1, -1, 2]) * k
    f_ = lambda t, c: (Q_in / V) * c_in + k_ * c[0] * c[1] - Q_out / V * c
    Jf_ = lambda t, c: np.matrix([[k_[0] * c[1] - Q_out / V, k_[0] * c[0], 0],
                                  [k_[1] * c[1], k_[1] * c[0] - Q_out / V, 0],
                                  [k_[2] * c[1], k_[2] * c[0], -Q_out / V]])

    y_euler_expl_ = euler_expl(f_, y0, t)
    y_euler_impr_ = euler_impr(f_, y0, t)
    y_AB2_ = twostep(f_, [], y0, [], t, a_AB2, b_AB2)
    y_AM2_ = twostep(f_, Jf_, y0, [], t, a_AM2, b_AM2)
    y_BDF2_ = twostep(f_, Jf_, y0, [], t, a_BDF2, b_BDF2)

    delta[0, i] = np.linalg.norm(y_euler_expl[:, j] - y_euler_expl_[:, j])
    delta[1, i] = np.linalg.norm(y_euler_impr[:, j] - y_euler_impr_[:, j])
    delta[2, i] = np.linalg.norm(y_AB2[:, j] - y_AB2_[:, j])
    delta[3, i] = np.linalg.norm(y_AM2[:, j] - y_AM2_[:, j])
    delta[4, i] = np.linalg.norm(y_BDF2[:, j] - y_BDF2_[:, j])

plt.figure()
plt.rcParams.update({'font.size': fs})
plt.plot(alpha, delta[0, :], 'r-o', alpha, delta[1, :], 'g-*', alpha, delta[2, :], 'b-*', alpha, delta[3, :], 'm-o',
         alpha, delta[4, :], 'k-*')
plt.legend(['Explicit Euler', 'Improved Euler', 'AB2', 'AM2', 'BDF2'], loc='best')
plt.title('Robustness study by changing $Q_{1,new} = \\alpha Q_1$')
plt.xlabel('$\\alpha$')
ylabel = ['$\|\Delta c_Z\|$', '$\|\Delta c_C\|$', '$\|\Delta c_P\|$']
plt.ylabel(ylabel[j])
plt.savefig(path + 'robustness.png')
