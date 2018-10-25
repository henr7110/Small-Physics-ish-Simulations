import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib import animation
fig, ax = plt.subplots()
L = 100
n_steps = 10000
Grid = np.zeros(L)
def SOC(Grid):
    pos = np.random.randint(L)
    Grid[pos] = 0
    Stable = False
    #Add grain
    Grid[np.random.randint(L-1)] += 1
    while not Stable:
        Stable = True
        for n in range(L):
            if Grid[n] >2:
                Stable=False
                Grid[n] = 0
                for i in range(2):
                    rand = np.random.randint(1,3)
                    if rand == 1 and n + 1 < L-1:
                        Grid[n+1] += 1
                    elif rand == 2 and n - 1 > 0:
                        Grid[n-1] += 1
    return np.copy(Grid)
def animate(i):
    line.set_ydata(animate.Grid)
    animate.Grid = SOC(animate.Grid)
animate.Grid = Grid

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_ylim(0,2)
line, = ax.plot(range(L),Grid)
anim = animation.FuncAnimation(fig, animate)
plt.show()
