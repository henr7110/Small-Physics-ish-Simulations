import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib import animation
L=20
p = 0
Grid = np.zeros((L+2,L+2))
Grid[1:-1,1:-1] = np.random.randint(2,size=(L,L))

def Periodic(Grid):
    Grid[:,-1]= Grid[:,1][:]
    Grid[-1,:]= Grid[1,:][:]
    Grid[:,0]= Grid[:,-2][:]
    Grid[0,:]= Grid[-2,:][:]
    return Grid

def NeighborCheck(Grid,px,py,val):
    value = val
    counter = 0
    if Grid[px+1,py] == value:
        counter += 1
    if Grid[px-1,py] == value:
        counter += 1
    if Grid[px,py+1] == value:
        counter += 1
    if Grid[px,py-1] == value:
        counter += 1
    return counter
# Periodic(Grid)
# plt.imshow(Grid)
# plt.show()

def SOC(Grid):

    px1,py1 = np.random.randint(1,L+1),np.random.randint(1,L+1)
    px2,py2 = np.random.randint(1,L+1),np.random.randint(1,L+1)
    while px2 == px1 and py1 == py2:
        px2,py2 = np.random.randint(1,L+1),np.random.randint(1,L+1)
    # print(px1,py1,px2,py2)
    if Grid[px1,py1] != Grid[px2,py2]:

        n1 = NeighborCheck(Grid,px1,py1,Grid[px1,py1])
        n2 = NeighborCheck(Grid,px2,py2,Grid[px1,py1])
        if n2 > n1:
            # print("channging")
            p1,p2 = np.copy(Grid[px1,py1]),np.copy(Grid[px2,py2])
            Grid[px1,py1] = p2
            Grid[px2,py2] = p1
        else:
            if np.random.random() < p:
                # print("channging")
                p1,p2 = np.copy(Grid[px1,py1]),np.copy(Grid[px2,py2])
                Grid[px1,py1] = p2
                Grid[px2,py2] = p1
    return Grid
def animate(i):
    if i %20 ==0:
        im.set_data(animate.Grid[1:-1,1:-1])
        ax1.set_title("itteration %d" % i)
    animate.Grid = Periodic(SOC(animate.Grid))
Periodic(Grid)
animate.Grid = Grid

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title("Empty")
im = plt.imshow(animate.Grid[1:-1,1:-1])
plt.colorbar()
anim = animation.FuncAnimation(fig, animate,interval=0)
plt.show()
