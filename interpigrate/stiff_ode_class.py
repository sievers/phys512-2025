import numpy as np
from scipy.integrate import solve_ivp
import time

def f(x,y,lives=[1,1e-5]):  #our decay function
    grad=0*y
    grad[0]=-y[0]/lives[0]
    grad[1]=y[0]/lives[0]-y[1]/lives[1]
    grad[2]=y[1]/lives[1]
    return grad


x=[0,1] #integrate from 0 to 1
y0=np.asarray([1,0,0.0])

t1=time.time()
ans_rk4=solve_ivp(f,x,y0)
t2=time.time()
ans_stiff=solve_ivp(f,x,y0,method='Radau')
t3=time.time()
print(t2-t1,t3-t2)
print('function evaluations: ',ans_rk4.nfev,ans_stiff.nfev)
