import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib import animation

n_steps = 1000
p = 0.44
L = 200
X = np.zeros((n_steps,L))
X[0,:] = np.random.randint(2,size=200)

def Perculate(i,X):
    #Birth downstreams
    for n in range(len(X[0,:])):
        if X[i-1,:][n]==1 and np.random.random() < p:
            X[i,:][n] = 1
    #Birth left and right
    for n in range(len(X[0,:])):
        if n-1 >0 and X[i-1][n-1] == 1 and np.random.random() <p:
            X[i,:][n] = 1

        if n+1 < L and X[i-1][n+1] == 1 and np.random.random() <p:
            X[i,:][n] = 1
    return X

def animate(i):
    im.set_data(animate.X)
    animate.X= Perculate(i+1,animate.X)
animate.X = X

fig = plt.figure()
ax1 = fig.add_subplot(111)
im = plt.imshow(X,aspect='auto')
anim = animation.FuncAnimation(fig, animate,interval=0)
plt.show()
