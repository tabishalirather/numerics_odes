import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from exercise import *
# from solution import *

#---------------------------------------------------------------------
# main program
#---------------------------------------------------------------------         
#path = "/home/student/result/" # YAPS path
path = "plots/" # <--- set private path for offline work, or just ignore it
plt.close('all')

w0 = np.array([1,0,0,1]) # initial condition w0 = [1,0,0,1] 
t0 = 0                   # integration interval [0,10]
te = 10

R = 7/9
#R = 4/3
S = 0.01
f_ = lambda t,w : f(t,w,R,S)
Jf_ = lambda t,w : Jf(t,w,R,S)
# Jf_ = []

h = 0.001                  
N = int((te-t0)/h)
t = np.linspace(t0,te,N+1)
w_ex = w_exact(t,R,S,w0)
print(w_ex)

#--------------------------------------------------------------------------
# apply distinct methods of Python's solve_ivp: RK45, Radau, BDF
#--------------------------------------------------------------------------
sol_RK45 = solve_ivp(f_,[t0,te],w0,method = 'RK45') # RK45
t_RK45 = sol_RK45.t 
w_RK45 = sol_RK45.y.T

sol_Radau = solve_ivp(f_,[t0,te],w0,method = 'Radau') # Radau
t_Radau = sol_Radau.t 
w_Radau = sol_Radau.y.T

sol_BDF = solve_ivp(f_,[t0,te],w0,method = 'BDF') # BDF
t_BDF = sol_BDF.t 
w_BDF = sol_BDF.y.T

fs = 12 # font size
plt.rcParams.update({'font.size': fs})
fig, axs = plt.subplots(2,2, figsize=(10, 8))
axs[0,0].plot(t_RK45,w_RK45[:,0],'r',t_Radau,w_Radau[:,0],'--b',t_BDF,w_BDF[:,0],'.-g',t,w_ex[:,0],'k')
axs[0,0].set_xlabel('t')
axs[0,0].set_ylabel('$y_1$')
axs[0,0].set_title('$y_1(t)$')
axs[0,0].legend(['RK45','Radau','BDF','anal. sol.'])

axs[0,1].plot(t_RK45,w_RK45[:,1],'r',t_Radau,w_Radau[:,1],'--b',t_BDF,w_BDF[:,1],'.-g',t,w_ex[:,1],'k')
axs[0,1].set_xlabel('t')
axs[0,1].set_ylabel('$y_2$')
axs[0,1].set_title('$y_2(t)$')
axs[0,1].legend(['RK45','Radau','BDF','anal. sol.'])

axs[1,0].plot(t_RK45,w_RK45[:,2],'r',t_Radau,w_Radau[:,2],'--b',t_BDF,w_BDF[:,2],'.-g',t,w_ex[:,2],'k')
axs[1,0].set_xlabel('t')
axs[1,0].set_ylabel('$v_1$')
axs[1,0].set_title('$v_1(t)$')
axs[1,0].legend(['RK45','Radau','BDF','anal. sol.'])

axs[1,1].plot(t_RK45,w_RK45[:,3],'r',t_Radau,w_Radau[:,3],'--b',t_BDF,w_BDF[:,3],'.-g',t,w_ex[:,3],'k')
axs[1,1].set_xlabel('t')
axs[1,1].set_ylabel('$v_2$')
axs[1,1].set_title('$v_2(t)$')
axs[1,1].legend(['RK45','Radau','BDF','anal. sol.'])
fig.tight_layout()
#fig.suptitle("Different methods of Python's solve_ivp")
plt.savefig(path + 'solve_ivp_methods.png')

# plot step sizes
fig, axs = plt.subplots(3, figsize=(8, 6))
# step sizes
h_RK45 = np.diff(t_RK45) 
h_Radau = np.diff(t_Radau) 
h_BDF = np.diff(t_BDF)
# average step sizes 
ha_RK45 = np.sum(h_RK45)/len(h_RK45)
ha_Radau = np.sum(h_Radau)/len(h_Radau)
ha_BDF = np.sum(h_BDF)/len(h_BDF)

axs[0].plot(t_RK45[0:-1],h_RK45,np.array([t0,te]),ha_RK45*np.array([1,1]),'r')
axs[0].set_xlabel('t')
axs[0].set_ylabel('h')
axs[0].set_title('RK45 step sizes, number of steps = ' + str(len(h_RK45)) +
                 ', average =  ' + str("{:.2e}".format(ha_RK45)) + ' (red line)')

axs[1].plot(t_Radau[0:-1],h_Radau,np.array([t0,te]),ha_Radau*np.array([1,1]),'--r')
axs[1].set_xlabel('t')
axs[1].set_ylabel('h')
axs[1].set_title('Radau step sizes, number of steps = ' + str(len(h_Radau)) +
                 ', average =  ' + str("{:.2e}".format(ha_Radau)) + ' (red line)')

axs[2].plot(t_BDF[0:-1],h_BDF,np.array([t0,te]),ha_BDF*np.array([1,1]),'--r')
axs[2].set_xlabel('t')
axs[2].set_ylabel('h')
axs[2].set_title('BDF step sizes, number of steps = ' + str(len(h_BDF)) + 
                  ', average =  ' + str("{:.2e}".format(ha_BDF)) + ' (red line)')
fig.tight_layout()
plt.savefig(path + 'stepsizes.png')



#--------------------------------------------------------------------------
# phase portrait y_1 vs. y2
#--------------------------------------------------------------------------


# How does this make any sense? The phase potraits when compared to the one in paper reference [1]
w79 = w_exact(t,7/9,0.3,w0)
w43 = w_exact(t,4/3,0.3,w0)

fig, axs = plt.subplots(1,2, figsize=(10, 10))
axs[0].plot(w79[:,0],w79[:,1])
axs[0].set_xlabel('$y_1$')
axs[0].set_ylabel('$y_2$')
axs[0].set_title('R = 7/9, S = 0.3')
axs[0].plot(w79[0,0], w79[0,1], 'ko', markersize=8, label='Start point  (t=0)')



axs[1].plot(w43[:,0],w43[:,1],'g')
axs[1].set_xlabel('$y_1$')
axs[1].set_ylabel('$y_2$')
axs[1].set_title('R = 4/3, S = 0.3')
axs[1].plot(w43[0,0],w43[0,1], 'ko', markersize=8, label='Start point (t=0)')
fig.tight_layout()
plt.legend()
plt.savefig(path + 'trajectory.png')


#--------------------------------------------------------------------------
# use a self-programmed solver (with fixed step size)
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
# collection of explicit Runge-Kutta methods
#--------------------------------------------------------------------------
# a = np.array([0]); B = np.array([0]); c = np.array([1]); txt = 'expl. Euler'                                # explicit Euler
# a = np.array([0,1/2]); B = np.array([[0,0],[1/2,0]]); c = np.array([0,1]); txt = 'impr. Euler'              # improved Euler 
# a = np.array([0,1]); B = np.array([[0,0],[1,0]]); c = np.array([1/2,1/2]); txt = 'Heun-2'                   # Heun-2    
a = np.array([0,2/3]); B = np.array([[0,0],[2/3,0]]); c = np.array([1/4,3/4]); txt = 'Ralston'              # Ralston
#a = np.array([0,1,2])/3; B = np.array([[0,0,0],[1,0,0],[0,2,0]])/3; c = np.array([1,0,3])/4; txt = 'Heun-3'  # Heun-3
#a = np.array([0,1,1,2])/2; B = np.array([[0,0,0,0],[1,0,0,0],[0,1,0,0],[0,0,1,0]])/2; c = np.array([1,2,2,1])/6; txt = 'classic RKM' # classical Runge-Kutta
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
# collection of diagonally implicit Runge-Kutta methods
#--------------------------------------------------------------------------
# a = np.array([1]); B = np.array([1]); c = np.array([1]); txt = 'impl. Euler'                                                        # implicit Euler
#a = np.array([0,1]); B = np.array([[0,0],[1/2,1/2]]); c = np.array([1/2,1/2]); txt = 'Trapezoidal rule'                              # Trapezoidal rule 
# z = 1/(2*np.sqrt(3)); a = np.array([1/2+z,1/2-z]); B = np.array([[1/2+z,0],[-2*z,1/2+z]]); c = np.array([1/2,1/2]); txt = 'Norsett'   # Norsett formula
# a = np.array([0,2/3]); B = np.array([[0,0],[1/3,1/3]]); c = np.array([1/4,3/4]); txt = 'Hammer & Hollingsworth'                     # Hammer & Hollingsworth
#a = np.array([0,1/2,1]); B = np.array([[0,0,0],[1/4,1/4,0],[0,1,0]]); c = np.array([1/6,2/3,1/6]); txt = 'Lobatto-IIIC*'             # Lobatto-IIIC*
#z = 2/np.sqrt(3) * np.cos(np.pi/18); a = np.array([(1+z)/2,1/2,(1-z)/2]); 
#B = np.array([[(1+z)/2,0,0],[-z/2,(1+z)/2,0],[1+z,-(1+2*z),(1+z)/2]]); c = np.array([1/(6*z**2),1-1/(3*z**2),1/(6*z**2)]); txt = 'Crouzeix-3' # Crouzeix's three-stage, 4th order
#--------------------------------------------------------------------------

w = runge_kutta(f_,Jf_,w0,t,a,B,c)       

fig, axs = plt.subplots(2,2)
axs[0,0].plot(t,w[:,0],'b',t,w_ex[:,0],'--r')
axs[0,0].set_xlabel('t')
axs[0,0].set_ylabel('$y_1$')
axs[0,0].set_title('$y_1(t)$')
axs[0,0].legend([txt,'anal. sol.'])

axs[0,1].plot(t,w[:,1],'b',t,w_ex[:,1],'--r')
axs[0,1].set_xlabel('t')
axs[0,1].set_ylabel('$y_2$')
axs[0,1].set_title('$y_2(t)$')
axs[0,1].legend([txt,'anal. sol.'])

axs[1,0].plot(t,w[:,2],'b',t,w_ex[:,2],'--r')
axs[1,0].set_xlabel('t')
axs[1,0].set_ylabel('$v_1$')
axs[1,0].set_title('$v_1(t)$')
axs[1,0].legend([txt,'anal. sol.'])

axs[1,1].plot(t,w[:,3],'b',t,w_ex[:,3],'--r')
axs[1,1].set_xlabel('t')
axs[1,1].set_ylabel('$v_2$')
axs[1,1].set_title('$v_2(t)$')
axs[1,1].legend([txt,'anal. sol.'])
fig.tight_layout()
#fig.suptitle("Self-programmed Runge-Kutta methods")
plt.savefig(path + 'Runge_Kutta_method.png')

#--------------------------------------------------------------------------
# collection of multistep methods
#--------------------------------------------------------------------------
# # Adams-Bashforth
# a = np.array([-1,1]); b = np.array([1]); txt = 'AB-1'                                           # AB-1
# a = np.array([0,-1,1]); b = np.array([-1,3])/2; txt = 'AB-2'                                    # AB-2
# a = np.array([0,0,-1,1]); b = np.array([5,-16,23])/12; txt = 'AB-3'                             # AB-3
# a = np.array([0,0,0,-1,1]); b = np.array([-9,37,-59,55])/24; txt = 'AB-4'                       # AB-4
# a = np.array([0,0,0,0,-1,1]); b = np.array([251,-1274,2616,-2774,1901])/720; txt = 'AB-5'       # AB-5

# # Adams-Moulton
# a = np.array([-1,1]); b = np.array([1,1])/2; txt = 'AM-1'                                       # AM-1
a = np.array([0,-1,1]); b = np.array([-1,8,5])/12; txt = 'AM-2'                                 # AM-2
# a = np.array([0, 0,-1,1]); b = np.array([1,-5,19,9])/24; txt = 'AM-3'                           # AM-3
# a = np.array([0,0,0,-1,1]); b = np.array([-19,106,-264,646,251])/720; txt = 'AM-4'              # AM-4
# a = np.array([0,0,0,0,-1,1]); b = np.array([27,-173,482,-789,1427,475])/1440; txt = 'AM-5'        # AM-5

# # Milne-Simpson
# a = np.array([-1,0,1]); np.array(b = [1,4,1])/3; txt = 'MS-2'                                   # MS-2
# a = np.array([0,0,-1,0,1]); b = np.array([-1,4,24,124,29])/90; txt = 'MS-4'                     # MS-4

# # Backward Differencing Formulas 
# a = np.array([-1,1]); b = np.array([0,1]); txt = 'BDF-1'                                        # BDF-1
#a = np.array([1/2,-2,3/2]); b = np.array([0,0,1]); txt = 'BDF-2'                                 # BDF-2
# a = np.array([-1/3,3/2,-3,11/6]); b = np.array([0,0,0,1]); txt = 'BDF-3'                        # BDF-3
# a = np.array([1/4,-4/3,3,-4,25/12]); b = np.array([0,0,0,0,1]); txt = 'BDF-4'                   # BDF-4
# a = np.array([-1/5,5/4,-10/3,5,-5,137/60]); b = np.array([0,0,0,0,0,1]); txt = 'BDF-5'          # BDF-5
#a = np.array([1/6,-6/5,15/4,-20/3,15/2,-6,49/20]); b = np.array([0,0,0,0,0,0,1]); txt = 'BDF-6'  # BDF-6

#--------------------------------------------------------------------------

w = k_step(f_,Jf_,w0,[],t,a,b) 

fig, axs = plt.subplots(2,2)
axs[0,0].plot(t,w[:,0],'b',t,w_ex[:,0],'--r')
axs[0,0].set_xlabel('t')
axs[0,0].set_ylabel('$y_1$')
axs[0,0].set_title('$y_1(t)$')
axs[0,0].legend([txt,'anal. sol.'])

axs[0,1].plot(t,w[:,1],'b',t,w_ex[:,1],'--r')
axs[0,1].set_xlabel('t')
axs[0,1].set_ylabel('$y_2$')
axs[0,1].set_title('$y_2(t)$')
axs[0,1].legend([txt,'anal. sol.'])

axs[1,0].plot(t,w[:,2],'b',t,w_ex[:,2],'--r')
axs[1,0].set_xlabel('t')
axs[1,0].set_ylabel('$v_1$')
axs[1,0].set_title('$v_1(t)$')
axs[1,0].legend([txt,'anal. sol.'])

axs[1,1].plot(t,w[:,3],'b',t,w_ex[:,3],'--r')
axs[1,1].set_xlabel('t')
axs[1,1].set_ylabel('$v_2$')
axs[1,1].set_title('$v_2(t)$')
axs[1,1].legend([txt,'anal. sol.'])
fig.tight_layout()
#fig.suptitle("Self-programmed multistep methods")
plt.savefig(path + 'k_step_method.png')




