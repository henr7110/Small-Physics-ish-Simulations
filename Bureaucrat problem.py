import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib import animation
L=50
Grid = np.zeros((L,L))
Avalanches = []
def SOC(Grid):
    #Add sand
    counter = 0
    #Make avalanche
    found = False
    for i in range(L):
        for j in range(L):
            if Grid[i,j] >= 4:
                found = True
                counter += 1
                Grid[i,j] = 0
                if not i+1 > L-1:
                    Grid[i+1,j] += 1
                    if not i-1 < 0:
                        Grid[i-1,j] += 1
                    if not j+1 > L-1:
                        Grid[i,j+1] += 1
                    if not j-1 < 0:
                        Grid[i,j-1] += 1
    if not found:
        Grid[24,24] += 1
    if counter != 0:
        Avalanches.append(counter)
    return Grid, Avalanches

def animate(i):
    im.set_data(animate.Grid)
    ax2.cla()
    ax2.hist(animate.Avalanches)
    animate.Grid,animate.Avalanches = SOC(animate.Grid)
animate.Grid = Grid
animate.Avalanches = Avalanches

fig = plt.figure()
ax1 = fig.add_subplot(121)
im = plt.imshow(Grid)
plt.colorbar()
plt.clim(0,4)
ax2 = fig.add_subplot(122)
hist = ax2.hist(Avalanches)
anim = animation.FuncAnimation(fig, animate,interval=0)
plt.show()
