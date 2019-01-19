from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate

# from plotly import offline as py
# import plotly.tools as tls
# py.init_notebook_mode()
# initialize grid,
width = 0.1
dxy = 0.01
t_steps = 50
dt = 0.01

#set grid
x = np.arange(-width,width,dxy)
y = np.arange(-width,width,dxy)

X,Y = np.meshgrid(x,y)

phi = np.exp( -((X-10)/10)**2  -((Y-10)/10)**2)
u = np.array([Y , -X])

#save phi to compare before and after simulation
beforephi = phi.copy()

fig = plt.figure()
#start calculation
for t in range(t_steps):

    phic = np.copy(phi)

    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, phi, color='b')
    plt.savefig("t=%d" %t)
    plt.cla()
    plt.clf()

    #backtrace position
    pastX, pastY = np.copy(X) - (dt * np.copy(Y)), np.copy(Y) - (dt * (- np.copy(X)))

    newfunc = interpolate.interp2d(x,y,phic)
    #Lookup X boundaries
    for i in range(len(pastX[0,:])):
        for j in range(len(pastX[:,0])):
            val = pastX[i,j]
            #Left boundary
            if val < -width:
                pastX[i,j] = -width + 0.0001
            #right boundary
            if val > width:
                pastX[i,j] = width-0.0001
    #Lookup Y boundaries
    for i in range(len(pastY[0,:])):
        for j in range(len(pastY[:,0])):
            val = pastY[i,j]
            #Left boundary
            if val < -width:
                pastY[i,j] = -width +0.0001
            #right boundary
            if val > width:
                pastY[i,j] = width-0.0001

    phi = [newfunc(pastX.ravel()[20*i:20*(i+1)],pastY.ravel()[20*i:20*(i+1)]) for i in range(len(X[:,0]))]

    X = pastX
    Y = pastY
#plot shit
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(np.array(X), np.array(Y), np.array(beforephi), color='b')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(np.array(X), np.array(Y), np.array(phi), color='b')
plt.show()

# plt.quiver(X,Y,u[0],u[1])
# plt.show()

# py.iplot(tls.mpl_to_plotly(plt.gcf()))
