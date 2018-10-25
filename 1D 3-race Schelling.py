import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib import animation
L=100
p = 0
Grid = np.zeros((L+2))
Grid[1:-1] = np.random.randint(3,size=L)
def Periodic(Grid):
    Grid[-1]= Grid[1]
    Grid[0]= Grid[-2]
    return Grid
Periodic(Grid)

def NeighborCheck(Grid,px,val):
    value = val
    counter = 0
    if Grid[px+1] == value:
        counter += 1
    if Grid[px-1] == value:
        counter += 1
    return counter

def SOC(Grid):
    p1,p2 = np.random.randint(1,L+1),np.random.randint(1,L+1)
    while p2 == p1:
        p2 = np.random.randint(1,L+1)
    # print(px1,py1,px2,py2)
    if Grid[p1] != Grid[p2]:
        n1 = NeighborCheck(Grid,p1,Grid[p1])
        n2 = NeighborCheck(Grid,p2,Grid[p1])
        if n2 > n1:
            # print("channging")
            p11,p22 = np.copy(Grid[p1]),np.copy(Grid[p2])
            Grid[p1] = p22
            Grid[p2] = p11
        else:
            if np.random.random() < p:
                # print("channging")
                p11,p22 = np.copy(Grid[p1]),np.copy(Grid[p2])
                Grid[p1] = p22
                Grid[p2] = p11
    return Grid
def animate(i):
    if i % 100 ==0:
        line.set_ydata(animate.Grid[1:-1])
        ax1.set_title("itteration %d" % i)
    animate.Grid = Periodic(SOC(animate.Grid))
Periodic(Grid)
animate.Grid = Grid

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title("Empty")
line, = ax1.plot(animate.Grid[1:-1])
anim = animation.FuncAnimation(fig, animate,interval=0)
plt.show()
