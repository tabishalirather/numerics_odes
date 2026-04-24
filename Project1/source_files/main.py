import numpy as np
import matplotlib.pyplot as plt
from exercise import zombie    # template for students


def zplot(t, x, title, fname):
  # path = "/home/student/result/" # YAPS path for storing plots
  path = "plots/"                        # local user path for storing plots
  fs = 15 # FontSize
  Tend = t[len(t)-1];
  # plt.figure("/i,ages")
  plt.figure()
  plt.rcParams.update({'font.size': fs})
  plt.plot(t, x[:,0],'b', t, x[:,1], '-g', t, x[:,2], '--k')
  plt.legend(['Humans','Zombies','Removed'],loc = 'best')
  plt.xlabel('Time t')
  plt.ylabel('Number')
  plt.title(title)
  plt.xlim(0.0, Tend)
  # plt.show()
  plt.savefig(path + fname)
  N = len(t)-1
  # print(f"len(t): {len(t)}: {N}")
  print('\n', title,'\n\t t = ',t[N],'\n\t #Humans = {:.2f}'.format(x[N,0]),'\n\t #Zombies = {:.2f}'.format(x[N,1]),'\n\t #Removed =  {:.2f}'.format(x[N,2]))

  return

params_max_error = 0.25 # uncertainty of parameters alpha and beta in percent

t, x = zombie(0,0,0)
print(f"t in __main__.py is: {t[99]}")
zplot(t,x,"Baseline scenario without intervention","pic_1.png")

t, x = zombie(1,0,0)
zplot(t,x,"Baseline scenario + Strategy 1","pic_2.png")
#
t, x = zombie(2,0,0)
zplot(t,x,"Baseline scenario + Strategy 2","pic_3.png")
#
t, x = zombie(1, -params_max_error, params_max_error)
zplot(t,x,"Worst case scenario + Strategy 1","pic_4.png")

t, x = zombie(2, -params_max_error, params_max_error)
zplot(t,x,"Worst case scenario + Strategy 2","pic_5.png")
