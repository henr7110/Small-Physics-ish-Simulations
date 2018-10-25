

"""Author: Henrik Pinholt
Created: 7. december 2017 kl. 23.07
Animates the Euler integration of the double pendulum equations as solved in the
Hamiltonian formulation of analytical mechanincs
"""
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import odeint
from scipy.optimize import broyden1
from matplotlib import animation

def dXdt(Theta2,Theta1,P1,P2):
    """Calculate derivatives dX/dt based on the Euler-Lagrange equations"""
    #Two shorthand constants
    C1 = (P1*P2*np.sin(Theta1-Theta2))/(l1*l2*(m1+m2*np.sin(Theta1-Theta2)**2))
    C2 = (np.sin(2*(Theta1-Theta2))) * (l2**2*m2*P1**2+l1**2*(m1+m2)*P2**2- \
           l1*l2*m2*P1*P2*np.cos(Theta1-Theta2)) \
           / (2*l1**2*l2**2*(m1+m2*np.sin(Theta1-Theta2)**2)**2)
    #Calc derivatives
    dTheta1dt = (l2*P1-l1*P2*np.cos(Theta1-Theta2)) \
                 /(l1**2*l2*(m1+m2*np.sin(Theta1-Theta2)**2))
    dTheta2dt = (l1*(m1+m2)*P2-l2*m2*P1*np.cos(Theta1-Theta2))\
                 /(l1*l2**2*m2*(m1+m2*np.sin(Theta1-Theta2)**2))
    dP1dt = -(m1+m2)*g*l1*np.sin(Theta1) - C1+C2
    dP2dt = -m2*g*l2*np.sin(Theta2)+C1-C2
    return ([dTheta1dt, dTheta2dt, dP1dt, dP2dt])
def remover(list,length):
    """Function that keeps lenght of list "length" long"""
    if len(list) > length:
        return list[-length:]
    else:
        return list
def animate(i):
    """Animation function to update the figure and run simulation"""
    #Calculate position of pendulum
    p1 = l1*np.array([np.sin(animate.Theta1),-np.cos(animate.Theta1)])
    p2 = p1 + l2*np.array([np.sin(animate.Theta2),-np.cos(animate.Theta2)])
    #Save position
    animate.p1x.append(p1[0])
    animate.p1y.append(p1[1])
    animate.p2x.append(p2[0])
    animate.p2y.append(p2[1])
    animate.p1x = remover(animate.p1x,200)
    animate.p1y = remover(animate.p1y,200)
    animate.p2x = remover(animate.p2x,200)
    animate.p2y = remover(animate.p2y,200)
    #Update plot
    line1.set_data([0,p1[0]],[0,p1[1]])
    circle1.center = p1
    line2.set_data([p1[0],p2[0]],[p1[1],p2[1]])
    line3.set_data(animate.p1x,animate.p1y)
    line4.set_data(animate.p2x,animate.p2y)
    circle2.center = p2
    #Get new data
    animate.Theta1,animate.Theta2,animate.P1,animate.P2 = \
    Update(animate.Theta1,animate.Theta2,animate.P1,animate.P2,animate.h)
def Update(Theta1,Theta2,P1,P2,h):
    """Euler integration procedure that numerically integrates based on local
    Taylor expansion to first order"""
    #calculate derivative
    result = dXdt(Theta2,Theta1,P1,P2)
    dTheta1dt,dTheta2dt,dP1dt,dP2dt = \
    result[0],result[1],result[2],result[3]
    #update pools
    Theta1 = Theta1 + h*(dTheta1dt)
    Theta2 = Theta2 + h*(dTheta2dt)
    P1 = P1 + h*(dP1dt)
    P2 = P2 + h*(dP2dt)
    return Theta1,Theta2,P1,P2

#Initialize values (it is here tha you tweak masses and lengths ect.)
l1,l2,m1,m2,g, = 4,2,1,1,9.84,
r1,r2,steps = m1*0.4,m2*0.4,100
Theta1_0, Theta2_0, P1_0, P2_0 = 0,0,50,50
animate.Theta1 = Theta1_0
animate.Theta2 = Theta2_0
animate.P1 = P1_0
animate.P2 = P2_0
animate.h = 0.01

#Initialize figure
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_xlim((-l1-l2,l1+l2))
ax1.set_ylim((-l1-l2,l1+l2))
#Calculate pendulum startpos
p1 = l1*np.array([np.sin(Theta1_0),-np.cos(Theta1_0)])
p2 = p1 + l2*np.array([np.sin(Theta2_0),-np.cos(Theta2_0)])
#Save
animate.p1x,animate.p1y = [p1[0]],[p1[1]]
animate.p2x,animate.p2y = [p2[0]],[p2[1]]
#Plot first instances of pendulum
circle1 = plt.Circle(p1,r1,fc="lightskyblue")
ax1.add_patch(circle1)
circle2 = plt.Circle(p2,r2,fc="firebrick")
ax1.add_patch(circle2)
line1, = ax1.plot([0,p1[0]],[0,p1[1]],c="black")
line2, = ax1.plot([p1[0],p2[0]],[p1[1],p2[1]],c="black")
line3, = ax1.plot(animate.p1x,animate.p1y,c="lightskyblue")
line4, = ax1.plot(animate.p2x,animate.p2y,c="firebrick")
#Animate
anim = animation.FuncAnimation(fig, animate,interval=0,frames=100)
plt.show()
