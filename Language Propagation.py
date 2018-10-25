import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib import animation
L=100
p = 0
Grid = np.zeros((L))
def SOC(Grid,status):
    if status == L/2-L/20:
        Grid[int(L/2)] += 1
        status = 0
    p1 = np.random.randint(1,L+1)
    if p1 > L/2:
        p2 = p1 - 1
    else:
        p2 = p1 + 1
    while p2 == p1:
        p2 = np.random.randint(1,L+1)
    if Grid[p1] != Grid[p2]:
        if Grid[p1] > Grid[p2]:
            Grid[p2] = np.copy(Grid[p1])
            status += 1
        else:
            Grid[p1] = np.copy(Grid[p2])
            status += 1
    return Grid,status
def animate(i):
    if i % 1 ==0:
        line.set_ydata(animate.Grid[1:-1])
        ax1.set_title("itteration %d" % i)
        ax1.set_ylim(min(Grid),max(Grid)+1)
    animate.Grid,animate.status = SOC(animate.Grid,animate.status)
animate.Grid = Grid
animate.status = L/2-L/20

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title("Empty")
line, = ax1.plot(animate.Grid[1:-1])
anim = animation.FuncAnimation(fig, animate,interval=0)
plt.show()
