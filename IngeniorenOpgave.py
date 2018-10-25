#Created 25. Oktober by Henrik Pinholt

#Simulation of the solution to the Prison guard riddle:
# There is a prison of N prisoners. The guard is nice and decides to give the
# prisoners a chance to get out:
# He puts all prisoners in isolation and draws them out at random to his office.
# At his office is two different coins. The prisoner has to flip one of them, and
# at first they start out as tails. At any point when in the office, you can
# proclaim that all prisoners has been in the office at least once. If true
# they all get set free.

#The answer goes like this:
#Designate a counter ot make the proclamation, and all other flip one of the coins
# as a dummy, but use the other coin to specify that they have been there

#Pseudocode:
#Make a list of N=100 prisoners, let the 0th be the counter.
#Initialize the coins to be 0
#On each update choose a random prisoner
#If he is the counter look at the coins and update counter if 1 don't if 0
#If he is prisoner, update coins to 1 if 0, if 0 update to 1 and let himself
#be chosen
import numpy as np
import matplotlib.pyplot as plt
class Prisoner:
    def __init__(self,n):
        self.picked = 0
        self.name = n

    def Picked(self,coin):
        if coin == 1:
            return 1
        elif coin == 0 and self.picked == 0:
            self.picked = 1
            #print("Prisoner %d: First time in" % self.name)
            return 1
        elif coin == 0 and self.picked == 1:
            return 0

class Counter:
    def __init__(self):
        self.count = 0

    def Picked(self,coin):
        if coin == 1:
            self.count += 1
            #print("Counter: Upping count")
            return 0
        if coin == 0:
            return 0
    def GetCount(self):
        return self.count

class PrisonGuard:
    def __init__(self,N):
        self.N = N
        self.prisoners = [Counter()]
        self.coin = 0
        for i in range(N-1):
            self.prisoners.append(Prisoner(i))

    def DragToOffice(self):
        prisoner = self.prisoners[np.random.randint(self.N)]
        self.coin = prisoner.Picked(self.coin)
        count = self.prisoners[0].GetCount()
        if count == self.N-1:
            return True,count
        else:
            return False,count
#Start sim
experiments = 1000
results = []
for i in range(experiments):
    if i % 100 == 0:
        print(i)
    Tim = PrisonGuard(100)
    done = False
    countbefore = 0
    itterations = 0
    while not done:
        # if itterations % 10000 == 0:
        #     print(itterations)
        done,countnow = Tim.DragToOffice()
        # if countbefore != countnow:
        #     #print(countnow,itterations)
        countbefore = np.copy(countnow)
        itterations += 1
    results.append(itterations)
mean = np.sum(results)/experiments
var = (np.var(results))**(1/2)
print("It takes on average %.01f visits to the Guard" % (mean))
print("The variance is %.01f " % (var))
plt.hist(results)
plt.axvline(x=mean,linestyle="c="black")
plt.axvline(x=mean+var,c="black")
plt.axvline(x=mean-var,c="black")
plt.show()
