import numpy as np #Modul der har en masse funktioner med matricer
import numpy.random as r #Modul der kan lave tilfældige tal
import matplotlib
matplotlib.use("Qt5Agg") # Sørger for at plotsene kommer op i egne vinduer
import matplotlib.pyplot as plt # bibliotek til at plotte med
plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
from matplotlib import animation # bibliotek til at plotte animere med
class Ladning():
    def __init__(self,pos,q,v=np.zeros(2),a=np.zeros(2),graenser=np.array([1,1])):
        """Ladning placeret på x,y"""
        self.pos,self.q = pos,q # ladning i enheder af elektronladningen
        self.v,self.a = np.array(v),np.array(a) #startv og a
        self.graenser = graenser # gem hvor elektronen må være
    def Kraft(self,andre,Q):
        """Beregner kraften baseret på kraften fra alle de andre"""

        afstande = np.sqrt(np.sum((self.pos-andre)**2,axis=1)) # Beregning af astanden til alle de andre
         #Culombs lov med 1/4pi*epsilon_0 = 1 og ladninger i enhed af elektronladningen
        krafter = -1*((self.q*Q*1/(afstande**2)) * ((andre-self.pos).T * (1/np.sqrt(np.sum((andre-self.pos)**2,axis=1))))).T
        kraftsum = np.sum(krafter,axis=0) #Den endelige kraft
        return kraftsum,afstande
    def Pot(self,andre,Q):
        """Beregner kraften baseret på kraften fra alle de andre"""

        afstande = np.sqrt(np.sum((self.pos-andre)**2,axis=1)) # Beregning af astanden til alle de andre
         #Culombs lov med 1/4pi*epsilon_0 = 1 og ladninger i enhed af elektronladningen
        Pot = np.sum(1*((self.q*Q*1/(afstande))))
        return Pot
    def Bevaeg(self,dt):
        """Bevaeg elektronen et tidsskridt baseret på dens nuværende hastighed og acceleration"""
        self.pos = (self.pos + dt*self.v + 0.5*dt**2*self.a)
    def OpdatV(self,dt,kraft):
        """Opdaterer hastigheden baseret på kraften på sig (massen er 1)"""
        self.v = self.v + (0.5*dt*(self.a+kraft))#-3e-4*self.v
        #Grænsebetingelse: bliv i kassen
        if self.pos[0] > self.graenser[0] and self.v[0] > 0: # højre side
            self.v[0] = -self.v[0]
        if self.pos[0] < -self.graenser[0] and self.v[0] < 0: #venstre side
            self.v[0] = -self.v[0]
        if self.pos[1] > self.graenser[1] and self.v[1] > 0: # top af kassen
            self.v[1] = -self.v[1]
        if self.pos[1] < -self.graenser[1] and self.v[1] < 0: #bund af kassen
            self.v[1] = -self.v[1]

    #Funktioner til at få elektronens position og ændre den
    def GetPos(self):
        return self.pos
    def GetV(self):
        return self.v
    def GetA(self):
        return self.a
    def SetPos(self,pos):
        self.pos = pos
    def SetV(self,v):
        self.v = v
    def SetA(self,a):
        self.pos = a
    def GetQ(self):
        return self.q

class Kasse():
    """Kasse til ladninger"""
    def __init__(self,pos,qs,dt=8e-3,v0s=[],a0s=[],graenser=1):
        """Initiér kasse"""
        self.graenser = graenser
        self.dt = dt
        #start ladninger
        if len(v0s) == 0:
            if len(a0s) == 0:
                self.ladninger = [Ladning(np.array(p),q) for p,q in zip(pos,qs)]
            else:
                self.ladninger = [Ladning(np.array(p),q,a=np.array(a)) for p,q,a in zip(pos,qs,a0s)]
        else:
            if len(a0s) == 0:
                self.ladninger = [Ladning(np.array(p),q,v=np.array(v)) for p,q,v in zip(pos,qs,v0s)]
            else:
                self.ladninger = [Ladning(np.array(p),q,v=np.array(v),a=np.array(a)) for p,q,v,a in zip(pos,qs,v0s,a0s)]

        self.pos = np.array([E.GetPos() for E in self.ladninger]) #list med elektronernes position
        self.Q = np.array([E.GetQ() for E in self.ladninger])
    def Opdater(self,retur=False):
        """Opdaterer positionen af alle elektronerne efter ét tidsskridt"""
        def dtfinder(mind,vmax):
            out = 5e-2/(1+(vmax/mind)**(1+1e-1))
            return out
        #Opdatér deres position baseret på deres nuværende acceleration og hastighed
        for E in self.ladninger:
            E.Bevaeg(self.dt)

        #Beregn kraften på hver elektron (Dette er også accelerationen hvis massen sættes til 1)
        krafter = []
        for i in range(len(self.ladninger)): #loop over index for alle elektroner
            ls = self.pos[np.arange(len(self.pos))!=i]
            qs = self.Q[np.arange(len(self.pos))!=i]
            K,d = self.ladninger[i].Kraft(ls,qs)
            krafter.append(K) #gem krafterne

        maxv = np.max([np.sqrt(E.GetV()[0]**2+E.GetV()[1]**2) for E in self.ladninger])
        krafter = np.array(krafter)
        mind = np.min(d)
        self.dt = dtfinder(mind,maxv)

        #Opdatér deres hastigheder baseret på denne
        for i in range(len(krafter)):
            self.ladninger[i].OpdatV(self.dt,krafter[i])
        #Gem deres positioner
        if not retur:
            self.pos = np.array([E.GetPos() for E in self.ladninger])
        else:
            self.pos = np.array([E.GetPos() for E in self.ladninger])
            return self.pos
    def GetPot(self,num):
        pot = np.zeros((num,num))
        minR = 1e-4
        normalize = True
        xs,ys = np.linspace(-1,1,num),np.linspace(-1,1,num)
        for i,x in enumerate(xs):
            for j,y in enumerate(ys):
                minr = np.min([np.sqrt(np.sum((p-np.array([x,y]))**2)) for p in self.pos])
                if minr > minR:
                    l = Ladning(np.array([x,y]),1)
                    if normalize:
                        pot[i,j] = l.Pot(self.pos,self.Q)*minr
                    else:
                        pot[i,j] = l.Pot(self.pos,self.Q)
        return pot.T[::-1,:]
    def Ballistisk(self,N,s=20,lines=0,field=0,live=True):
        """Animer elektronernes bevægelse"""
        #Initiér animationen
        if live:
            self.lines=lines
            fig,ax = plt.subplots(1,1,figsize=(10,10))
            cdict = {True:"g",False:"b"}
            colors = [cdict[q>0] for q in self.Q]
            scat = ax.scatter(self.pos[:,0],self.pos[:,1],s=s,color=colors) # plotter x og y for alle elektronerne
            ax.set(xlabel="x",ylabel="y",ylim=(-self.graenser,self.graenser),xlim=(-self.graenser,self.graenser)) # Sæt hvordan plottets grænser er og aksetitler
            ax.set_title(f"Itteration nr. {-1}")
            if field != 0:
                I = ax.imshow(self.GetPot(field),extent=[-1,1,-1,1],cmap="coolwarm")
            if self.lines != 0:
                self.paths = self.pos
                lines = []
                for row,c in zip(self.paths,colors):
                    x,y = [row[i] for i in range(len(row)) if i % 2==0],[row[i] for i in range(len(row)) if i % 2==1]
                    lines.append(ax.plot(x,y,color=c,linewidth=2))
            def Frame(i):
                """Funktion der tegner den nye position af elektronerne"""
                try:
                    self.Opdater() #Opdatér positioner
                except:
                    raise EnvironmentError("Something went wrong")
                scat.set_offsets(self.pos) #Tegn de nye positioner på plottet
                if field != 0:
                    I.set_array(self.GetPot(field))
                if self.lines != 0:
                    self.paths = np.hstack((self.paths,self.pos))
                    for row,line in zip(self.paths,lines):
                        x,y = [row[i] for i in range(len(row)) if i % 2 ==0],[row[i] for i in range(len(row)) if i % 2==1]
                        if len(x) > self.lines:
                            line[0].set_data(x[-self.lines:],y[-self.lines:])
                        else:
                            line[0].set_data(x,y)
                ax.set_title(f"Itteration nr. {i}, dt={self.dt:4.2E}") #Skriv hvor langt den er
                return scat,lines

            # Set up formatting for the movie files
            ani = animation.FuncAnimation(fig,Frame,interval=0,frames=N,repeat=False) #Animér i N frames
            plt.show() #vis animationen
        else:
            #run sim
            time =[0]
            pos = [self.pos]
            felt = [self.GetPot(field)]
            for i in range(1,N):
                if i % int(N/10) == 0:
                    print(f"{10*i/int(N/10)}% done")
                time.append(time[i-1]+self.dt)
                self.Opdater()
                pos.append(self.pos)
                felt.append(self.GetPot(field))

            #hent de rigtige punkter i lige intervaller
            timeslice=np.max(np.array(time[1:])-np.array(time[:-1]))
            last = 0
            posplot = []
            timeplot = []
            if field != 0:
                fieldplot = []
            for i in range(N):
                if np.abs(time[i]-last) > timeslice:
                    last = time[i]
                    posplot.append(pos[i])
                    if field != 0:
                        fieldplot.append(felt[i])
                    timeplot.append(time[i])
            #initiér plot
            self.lines=lines
            fig,ax = plt.subplots(1,1,figsize=(10,10))
            cdict = {True:"g",False:"b"}
            colors = [cdict[q>0] for q in self.Q]
            scat = ax.scatter(pos[0][:,0],pos[0][:,1],s=s,color=colors) # plotter x og y for alle elektronerne
            ax.set(xlabel="x",ylabel="y",ylim=(-self.graenser,self.graenser),xlim=(-self.graenser,self.graenser)) # Sæt hvordan plottets grænser er og aksetitler
            ax.set_title(f"Time={time[0]:4.2E}")
            if field != 0:
                I = ax.imshow(felt[0],extent=[-1,1,-1,1],cmap="coolwarm")
            if self.lines != 0:
                self.paths = pos[0]
                lines = []
                for row,c in zip(self.paths,colors):
                    x,y = [row[i] for i in range(len(row)) if i % 2==0],[row[i] for i in range(len(row)) if i % 2==1]
                    lines.append(ax.plot(x,y,color=c,linewidth=2))
            #definér updateringsfunktion
            def Frame(i):
                """Funktion der tegner den nye position af elektronerne"""
                scat.set_offsets(posplot[i]) #Tegn de nye positioner på plottet
                if field != 0:
                    I.set_array(fieldplot[i])
                if self.lines != 0:
                    self.paths = np.hstack((self.paths,posplot[i]))
                    for row,line in zip(self.paths,lines):
                        x,y = [row[i] for i in range(len(row)) if i % 2 ==0],[row[i] for i in range(len(row)) if i % 2==1]
                        if len(x) > self.lines:
                            line[0].set_data(x[-self.lines:],y[-self.lines:])
                        else:
                            line[0].set_data(x,y)
                ax.set_title(f"T = {timeplot[i]:4.2E}") #Skriv hvor langt den er
                return scat,lines
            # Set up formatting for the movie files
            ani = animation.FuncAnimation(fig,Frame,interval=10,frames=len(timeplot),repeat=False) #Animér i N frames
            plt.show() #vis animationen
    def Retarded(self,N):
        run = [self.Opdater(retur=True) for i in range(N)]
        pos,dts = [r[0] for r in run],[r[1] for r in run]
        t,ts = 0,[]
        for dt in dts:
            t += dt
            ts.append(np.copy(t))

        plt.plot(ts)
        plt.show()
    def FeltPlot(self,s=10,minR=1e-4,surface=False,normalize=False):
        """Plot feltlinjer i kassen, skaleret med mindste afstand til ladning"""
        from mpl_toolkits.mplot3d import Axes3D
        ex,ey,px,py = [],[],[],[]
        pot = np.zeros((50,50))
        xs,ys = np.linspace(-1,1),np.linspace(-1,1)
        for i,x in enumerate(xs):
            for j,y in enumerate(ys):
                minr = np.min([np.sqrt(np.sum((p-np.array([x,y]))**2)) for p in self.pos])
                if minr > minR:
                    l = Ladning(np.array([x,y]),1)
                    felt = l.Kraft(self.pos,self.Q)[0]*(minr**2)
                    if normalize:
                        pot[i,j] = l.Pot(self.pos,self.Q)*minr
                    else:
                        pot[i,j] = l.Pot(self.pos,self.Q)
                    px.append(x)
                    py.append(y)
                    ex.append(felt[0])
                    ey.append(felt[1])
        if not surface:
            fig,ax = plt.subplots(1,1,figsize=(10,10))
            I = ax.imshow(pot.T[::-1,:],extent=[-1,1,-1,1],cmap="coolwarm")
            cbar = fig.colorbar(I)
            cbar.set_label("Potentiale")
            ax.quiver(px,py,ex,ey,pivot="middle",cmap="coolwarm")
            cdict = {True:"g",False:"b"}
            colors = [cdict[q>0] for q in self.Q]
            scat = ax.scatter(self.pos[:,0],self.pos[:,1],s=s,color=colors) # plotter x og y for alle elektronerne
            ax.set(xlabel="x",ylabel="y",ylim=(-1,1),xlim=(-1,1)) # Sæt hvordan plottets grænser er og aksetitler
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111,projection="3d")
            x,y = np.meshgrid(xs,ys)
            cdict = {True:"g",False:"b"}
            colors = [cdict[q>0] for q in self.Q]
            surf = ax.plot_surface(x,y,pot.T,cmap="coolwarm")
            for xs,ys,c in zip(self.pos[:,0],self.pos[:,1],colors):
                ax.scatter3D(xs,ys,np.max(pot)+0.5,color=c,s=s)
            ax.set(xlabel="x",ylabel="y",zlabel="Potentiale",ylim=(-1,1),xlim=(-1,1)) # Sæt hvordan plottets grænser er og aksetitler
        plt.show()
