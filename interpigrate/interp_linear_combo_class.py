import numpy as np
from matplotlib import pyplot as plt
plt.ion()

def f(x):
    return 1/x


x=np.linspace(1,2,41)
y=f(x)
ans=np.log(x[-1])-np.log(x[0]) #analytic answer for log(x)

dx=x[1]-x[0] #the spacing of our points
myans=(0.5*(y[0]+y[-1])+np.sum(y[1:-1]))*dx
print('got ',myans,' expected ',ans,' error: ',myans-ans)

#if I use some value of dx, I get ans+eps*dx**2+...  int(dx)
#if I use 2 dx, I get ans + eps*(2dx)*2 = ans + 4 eps dx^2 = int(2dx)
#4 * int(dx)-int(2dx) = 4*ans + 4*eps*dx**2 - (ans+4*eps*dx^2) = 3*ans+...(higher)
#so I can use ans=(4*int(dx)-int(2dx))/3

int1=myans
int2=(0.5*(y[0]+y[-1])+np.sum(y[2:-2:2]))*(2*dx)
print(int1,int2)
new_ans=(4*int1-int2)/3
print(new_ans,new_ans-ans)

#(0.5*y[0]+y[1]+y[2]+y[3]+...)*dx for int1, (0.5*y[0]+y[2]+y[4]...)*2dx for int2
#(4*int1-int2)/3 = (2*y[0]+4y[1]+4y[2]...)dx - (y[0]+2y[2]+2y[4]...)*dx
#goes to (y[0]+4y[1]+2y[2]+4y[3]+2y[4]+...+y[-1])*dx/3
check=(y[0]+y[-1]+4*np.sum(y[1:-1:2])+2*np.sum(y[2:-2:2]))*dx/3
print('alternate answer: ',check)
