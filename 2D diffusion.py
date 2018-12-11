"""
Author: Henrik Pinholt
11/12-2018
Diffusion simulation in 2D
"""
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import colorsys
class RandomWalker():
    def __init__(self,x,y,steplength):
        """Init positions"""
        self.x,self.y,self.steplength = x,y,steplength
        self.xlist,self.ylist = [x],[y]
    #Interaction functions
    def SetX(self,x):
        self.x = x
        self.xlist.append(x)
    def SetY(self,y):
        self.y = y
        self.ylist.append(y)
    def GetStepLength(self):
        return self.steplength
    def GetX(self):
        return self.x
    def GetY(self):
        return self.y
    def GetXlist(self):
        return self.xlist
    def GetYlist(self):
        return self.ylist
    #Run functions
    def Update(self):
        """
        Updates position based on random direction and steplength
        """
        import numpy as np
        import numpy.random as r
        #Pick random direction in [0,2pi]
        dir = r.random()*2*np.pi
        #Get new x,y in that direction
        xnew = self.x + np.cos(dir)*self.steplength
        ynew = self.y + np.sin(dir)*self.steplength
        #Update position
        self.SetX(xnew)
        self.SetY(ynew)
    #Plot functions
    def Draw(self,ax,color,linewidth,size):
        ax.plot(self.xlist,self.ylist,linestyle="-",linewidth=linewidth) #plot path
        ax.plot(self.x,self.y,"o",markersize=size)
class CenterGrid():
    def __init__(self,N_walkers,size):
        """Initialize grid with N_walkers rdm walkers in center and size (x,y)"""
        self.lenx,self.leny = size[0],size[1]
        self.steplength = 1e-2 #1e-4 of diagonal size GetStepLength
        self.walkers = [RandomWalker(0,0,self.steplength) for i in range(N_walkers)]
    #Simulation functions
    def Run(self,itterations):
        """update walkers itteration times and save result"""
        for i in range(itterations):
            for walker in self.walkers:
                walker.Update()
    #Plotting functions
    def PlotPaths(self,linescale,dotscale):
        N = len(self.walkers)
        R = len(self.walkers[0].GetXlist())
        density =  float(N)/(R/self.steplength)
        HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in range(N)]
        RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
        fig,ax = plt.subplots(figsize=(10,6))
        for walker,col in zip(self.walkers,RGB_tuples):
            walker.Draw(ax,col,linescale*density,dotscale*density)
        ax.set(xlabel="x",ylabel="y",xlim=(-self.lenx,self.lenx),ylim=(-self.leny,self.leny))
        ax.set_title(f"Status after {R-1} runs of {N} walkers")
        plt.show()
    def PlotEnds(self):
        N = len(self.walkers)
        R = len(self.walkers[0].GetXlist())
        density =  float(N)/(R/self.steplength)
        HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in range(N)]
        RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
        xends,yends = [w.GetX() for w in self.walkers],[w.GetY() for w in self.walkers]
        fig, ax = plt.subplots(figsize=(10, 6))
        counts, xedges, yedges, im = ax.hist2d(xends, yends, bins=[60, 40], range= [[-self.lenx, self.lenx], [-self.leny, self.leny]], cmin=1)
        fig.colorbar(im) # ticks=[-1, 0, 1]
        ax.set_title(f"Status after {R-1} runs of {N} walkers")
        plt.show()
    def AnimateWalkers(self,linescale):
        import matplotlib.animation as animation
        N = len(self.walkers)
        R = len(self.walkers[0].GetXlist())
        density =  float(N)/(R/self.steplength)
        fig = plt.figure(figsize=(10,6))
        HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in range(N)]
        RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
        plt.xlim(-self.lenx, self.lenx)
        plt.ylim(-self.leny, self.leny)
        plt.xlabel('x')
        plt.ylabel("y")
        plt.title(f"Status after 0 runs of {N} walkers")
        lines = [plt.plot(w.GetXlist()[0],w.GetYlist()[0],linewidth=linescale*density,
                 color=c) for w,c in zip(self.walkers,RGB_tuples)]
        def animate(i):
            if i == 0:
                pass
            else:
                for walker,line in zip(self.walkers,lines):
                    line[0].set_data(walker.GetXlist()[:i],walker.GetYlist()[:i])
                plt.title(f"Status after {i} runs of {N} walkers")
        ani = matplotlib.animation.FuncAnimation(fig, animate,frames=R-2,interval=0)
        plt.show()


master = CenterGrid(100,(1,1))
master.Run(2000)
master.PlotEnds()
master.PlotPaths(1e3,3e3)
master.AnimateWalkers(1e3)
