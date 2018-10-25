import matplotlib.pyplot as pyplot
import skfmm as sk
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
# from plotly import offline as py
# import plotly.tools as tls
# py.init_notebook_mode()
# initialize grid,
width = 1
dxy = 0.1
print(width / dxy)
t_steps = 100
epsilon = 0.0001
k_max = 1. / dxy
dt = dxy / (2 * k_max)


a = np.array([[1,2,3,4,5,6],[7,8,9,10,11,12],[13,14,15,16,17,18],[19,20,21,22,23,24],[25,26,27,28,29,30]])

#define stencil derivatives
def Dx(phi, dxy):
    """central derivative of entire numpy field (with ghost nodes)
    in x-direction """
    calc = (phi[:,2:] - phi[:,:-2]) / (2*dxy)
    return(calc[1:-1,:])

def Dy(phi, dxy):
    """central derivative of entire numpy field (with ghost nodes)
    in y-direction (j is 0 in top of grid and highest in bottom to define
    Dx and Dy symmetrically)"""
    calc = (phi[2:,:] - phi[:-2,:]) / (2*dxy)
    return(calc[:,1:-1])

def Dxx(phi,dxy):
    """central second order derivative of entire numpy field (with ghost nodes)
    in x-direction """
    calc = (phi[:,2:] + phi[:,:-2] - (2 * phi[:,1:-1])) / (dxy**2)
    return(calc[1:-1,:])

def Dyy(phi,dxy):
    """central second order derivative of entire numpy field (with ghost nodes)
    in y-direction """
    calc = (phi[2:,:] + phi[:-2,:] - (2 * phi[1:-1,:])) / (dxy**2)
    return(calc[:,1:-1])

def Dxy(phi,dxy):
    """central second order derivative of entire numpy field (with ghost nodes)
    in xy-direction """
    return((phi[2:,2:] - phi[:-2,2:] - phi[2:,:-2] + phi[:-2,:-2]) / (4 * dxy**2))

#set grid
x = np.arange(-width, width + dxy,dxy)

y = np.arange(-width, width + dxy,dxy)

X,Y = np.meshgrid(x,y)

template = np.zeros((len(x), len(y)))
ones = np.ones((int(len(x)/2),int(len(x)/2)))
template[int(len(x)/4):int(len(x)/4)+len(ones),int(len(x)/4):int(len(x)/4)+len(ones)] = ones
phi = sk.distance(template)
u = np.array([Y , -X])
plt.imshow(phi)
plt.show()
#save phi to compare before and after simulation
beforephi = phi.copy()

#make map from X-value/Y-value to list index:

nmap = {}
declen = len(str(dxy-int(dxy))[1:]) - 1
if type(width) == float:
    dimlen = len(str(width-int(width))[1:]) - 2
if type(width) == int:
    dimlen = len(str(width))
roundval = declen + dimlen

list = []
for n,val in zip(range(len(X[0,:])),X[0,:]):
    nmap[round(val,roundval)] = n
    list.append(val)

fig = plt.figure()
#start calculation
for t in range(t_steps):

    oldphi = np.copy(phi)
    if t%5 == 0:
        print(t)
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, phi, color='b')
        plt.savefig("t=%d" %t)
        plt.cla()
        plt.clf()

    #calculate k_ij
    g = np.sqrt(Dx(oldphi, dxy)**2 + Dy(oldphi, dxy)**2)
    #implement g_ij min value
    g[g <= 0.5] = 1

    k = ((Dx(oldphi, dxy))**2 * Dyy(oldphi, dxy) + \
         (Dy(oldphi, dxy))**2 * Dxx(oldphi, dxy) - \
         2 * Dxy(oldphi, dxy) * Dx(oldphi, dxy) * Dy(oldphi, dxy)) / (g**3)

    #clamp k_ij
    k[k > k_max] = k_max
    k[k < -k_max] = -k_max

    kgrid = np.zeros((len(X[0,:]),len(Y[:,0])))
    kgrid[1:-1,1:-1] = k
    phi = oldphi + kgrid * dt

#plot shit
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, beforephi, color='b')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, phi, color='b')
plt.show()

# plt.quiver(X,Y,u[0],u[1])
# plt.show()

# py.iplot(tls.mpl_to_plotly(plt.gcf()))
