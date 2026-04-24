import numpy as np
import matplotlib.pyplot as plt
from exercise import zombie    # template for students
#from solution import zombie   # solution
from main import zplot

def zplot_combined(results, fname):
  path = "plots/"
  fs = 12

  fig, axes = plt.subplots(2, 3, figsize=(18, 10))
  axes = axes.flatten()

  plt.rcParams.update({'font.size': fs})

  for i, (title, t, x) in enumerate(results):
    ax = axes[i]

    ax.plot(t, x[:, 0], 'b', label="Humans")
    ax.plot(t, x[:, 1], '-g', label="Zombies")
    ax.plot(t, x[:, 2], '--k', label="Removed")

    ax.set_title(title)
    ax.set_xlabel("Time t")
    ax.set_ylabel("Number")
    ax.set_xlim(0.0, t[-1])
    ax.legend(loc="best", fontsize=8)
    ax.grid(True)

  # remove unused last subplot (since 5 plots only)
  fig.delaxes(axes[-1])

  plt.tight_layout()
  plt.savefig(path + fname)
  plt.close()

param_error_max = 0.25
results = []

t, x = zombie(0, 0, 0)
results.append(("Baseline no intervention", t, x))
zplot(t, x, "Baseline scenario without intervention", "pic_1.png")

t, x = zombie(1, 0, 0)
results.append(("Baseline Strategy 1", t, x))
zplot(t, x, "Baseline scenario + Strategy 1", "pic_2.png")

t, x = zombie(2, 0, 0)
results.append(("Baseline Strategy 2", t, x))
zplot(t, x, "Baseline scenario + Strategy 2", "pic_3.png")

t, x = zombie(1, -param_error_max, param_error_max)
results.append(("Worst case scenario + Strategy 1", t, x))
zplot(t, x, "Worst case scenario + Strategy 1", "pic_4.png")

t, x = zombie(2, -param_error_max, param_error_max)
results.append(("Worst case scenario + Strategy 2", t, x))
zplot(t, x, "Worst case scenario + Strategy 2", "pic_5.png")

zplot_combined(results, "pic_all_combined.png")