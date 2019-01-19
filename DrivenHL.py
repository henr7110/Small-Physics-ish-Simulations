import matplotlib
matplotlib.use('Qt5Agg')
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
#Set initial variables
dt,f,px,py=1e-2,4,0.3,0
fig, ax = plt.subplots(2,1,figsize=(8,12))
#Make initial plot of the velocity field
x,y = np.linspace(-1,1,25),np.linspace(-1,1,25)
X, Y = np.meshgrid(x,y)
def GetUV(X,Y,t):
    U = Y
    V = -X - 1e-2*Y +np.cos(f*t)# Det her er snyd....:D
    return U,V
U,V = GetUV(X,Y,0)
Q = ax[0].quiver(X, Y, U, V, color='r',pivot='mid')
ax[0].axhline(y=0,linestyle="--",color="b",linewidth=1)
ax[0].axvline(x=0,linestyle="--",color="b",linewidth=1)
ax[0].set(xlim=[-1,1],ylim=[-1,1])
ax[0].set_ylabel(ylabel=r"$\dot{x}$",fontsize=25)
ax[0].set_title("Driven Harmonic Oscillator",fontsize=25)
#Plot the future path of the ball for overview
futurex,futurey = [px],[py]
tval=0
for x,y,i in zip(futurex,futurey,range(1000)):
    U,V = GetUV(x,y,tval)
    tval += dt
    futurex+=[x+U*dt]
    futurey+=[y+V*dt]
ax[0].plot(futurex,futurey,"g",linewidth=2)
path, = ax[0].plot(px,py,"k",linewidth=2)
scat = ax[0].scatter(px,py,color="r",s=100)
#Plot the spring setup
ax[1].axvline(x=1,linestyle="--",color="b",linewidth=1)
ax[1].add_patch(plt.Circle([1.75,0.2],radius=0.2))
bar, = ax[1].plot([1+px,1.75+0.2*np.cos(0)],[0.2,0.2+0.2*np.sin(0)],"k",linewidth=5)
ax[1].set(xlim=[0,2],ylim=[0,1])
ax[1].set_xlabel("x",fontsize=25)
omega = np.pi*2*4/(1+px)
x = np.linspace(0,1+px,200,endpoint=False)
spring, = ax[1].plot(x,0.2+0.2*np.sin(omega*x),"k-")
ball = ax[1].scatter([1+px],[0.2],color="red",s=400)

def animate(num, Q,scat):
    """Updates the plots by a time step dt"""
    animate.t += dt
    px,py = np.copy(animate.px),np.copy(animate.py)
    U,V = GetUV(X,Y,animate.t)
    Up,Vp  = GetUV(px,py,animate.t)
    animate.px,animate.py = px+Up*dt,py+Vp*dt
    bar.set_data([1+px,1.75+0.2*np.cos(f*animate.t)],[0.2,0.2+0.2*np.sin(f*animate.t)])
    animate.pathx += [animate.px]
    animate.pathy += [animate.py]
    omega = np.pi*2*4/(1+animate.px)
    Q.set_UVC(U,V)
    x = np.linspace(0,1+animate.px,200,endpoint=False)
    spring.set_data(x,0.2+0.2*np.sin(omega*x))
    ball.set_offsets([1+animate.px,0.2])
    path.set_data(animate.pathx,animate.pathy)
    scat.set_offsets(np.array([animate.px,animate.py]).T)
    return Q,bar,spring,ball,path,scat
#Save internal animation params
animate.px = px
animate.py = py
animate.pathx = [px]
animate.pathy = [py]
animate.t = 0
#run dat shit
anim = animation.FuncAnimation(fig, animate, fargs=(Q, scat),
                               interval=0, blit=True)
plt.show()
