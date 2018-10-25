import numpy as np
import matplotlib
# matplotlib.use('Qt5Agg')
from numpy.random import randint,random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
def RandomAssigner():
    return -0.5 if random() < 0.5 else 0.5
def UpAssigner():
    return 0.5
def Energy(Grid,J,h):
    Energy = 0
    for i in range(Grid.shape[0]):
        for j in range(Grid.shape[1]):
            Energy += h*Grid[i,j]
            for a in range(Grid.shape[0]):
                for b in range(Grid.shape[1]):
                    if a > i and b > j:
                        Energy += -J*Grid[i,j]*Grid[a,b]
    return Energy/(Grid.shape[0]*Grid.shape[1])
def Flipper(l1,neighbors,J,h,T):
    """calculates energy, evaluates min(1,exp(-betaDeltaE)) and decides to flip
    or not and returns state"""
    dE = -J*l1*sum(neighbors) - h*(sum(neighbors)+l1)
    if dE < 0 or np.random.random() < np.exp(-dE/T):
        return -1*l1
    #print (p_i, E_1-E_0,"not flipped")
    return l1
# Ts = np.linspace(0.2,0.6,20)
sims = []
# for T in Ts:
Ts = np.linspace(0.5,4,20)
#initialize lattice
for T,n in zip(Ts,range(len(Ts))):
    print(n)
    kb = 1.38064852*10e-23
    J=-1
    h=1
    n_steps= 100000
    N=400*400
    anim = False
    Grid = np.zeros((int(N**(1/2)),int(N**(1/2))))
    #fig = plt.figure()
    #Populate with random numbers
    for i in range(int(N**(1/2))):
        for j in range(int(N**(1/2))):
            Grid[i,j] = RandomAssigner()
    if anim:
        im = plt.imshow(Grid.copy(),animated=True)
    #RUN SIMULATION
    #run n_steps sim
    if anim:
        ims = [[im]]
    magnetization = []
    E = []
    for n in range(n_steps):
        # if n % int(n_steps/10) == 0:
        #     print (str(float(n/n_steps)*100) + " % done")
        #itterate over lattice and flip spins

        i,j=randint(int(N**(1/2))),randint(int(N**(1/2)))
        #grab rightmost and downmost neighbors (periodic boundary)
        if i == len(Grid)-1:
            i = -1
        if j == len(Grid)-1:
            j = -1
        neighbors = [Grid[i+1,j],Grid[i-1,j],Grid[i,j+1],Grid[i,j-1]]
        flipper = Flipper(Grid[i,j],neighbors,J,h,T)
        Grid[i,j] = flipper
        #log shit
        magnetization.append(np.sum(Grid)/N)
        # if n % 1000 and n>20000 ==0:
        #     E.append(Energy(Grid,J,h))
        if n % int(n_steps/10) == 0 and anim:
            ims.append([plt.imshow(Grid.copy(),animated=True)])
    sims.append(magnetization[:])
m = []
for i in sims:
    m.append(np.mean(i[:90000]))
# for a in range(len(sims)):
#     plt.plot(range(len(sims[a])),sims[a],label=str(Ts[a]))
#     plt.xlabel("T (J^-1)")
#     plt.ylabel("<m>/N")
# plt.legend()
plt.plot(Ts,m,"o")
plt.xlabel("T (J^-1)")
plt.ylabel("<m>/N")
plt.title("J=1,h=1")
# f, (ax1, ax2) = plt.subplots(2, 1, sharey=True)
# ax1.plot(range(len(magnetization)),magnetization,label="Magnetization")
# plt.xlabel("n")
# plt.ylabel("Magnetization")
# ax2.plot(range(len(E)),E,label="Energy")
# plt.xlabel("n")
# plt.ylabel("Energy (J^-1)")
# plt.show()
if anim:

    print ("animating")
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                    repeat_delay=200)
    mywriter = animation.FFMpegWriter()
    ani.save("dynamic_images.mp4",writer=mywriter)
plt.savefig("convergenceN")

# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# X,Y = np.meshgrid(Ts, range(len(magnetization)))
# surf = ax.plot_surface(X, Y, np.array(sims).T, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)
# ax.set_xlabel("Temperature (J)")
# ax.set_ylabel("Steps")
# ax.set_zlabel("Magnetization")
# fig.colorbar(surf, shrink=0.5, aspect=5)
# plt.show()
