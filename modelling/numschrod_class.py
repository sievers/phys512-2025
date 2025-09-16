import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import eigh_tridiagonal #faster than eigh on a full matrix
import time

#set up x points
x=np.linspace(-10,10,2001)
dx=x[1]-x[0]

V=0.5*x**2
V=V+50*np.exp(-0.5*x**2/0.1**2)

diag=1/dx**2+V
off_diag=np.zeros(len(diag)-1)-0.5/dx**2

t1=time.time()
e,v=eigh_tridiagonal(diag,off_diag)
t2=time.time()
print('tridiag: ',t2-t1)

n=len(x)
mat=np.zeros([n,n])
for i in range(n):
    mat[i,i]=V[i]+1/dx**2
for i in range(n-1):
    mat[i,i+1]=-0.5/dx**2
    mat[i+1,i]=-0.5/dx**2

t1=time.time()
e2,v2=np.linalg.eigh(mat)
t2=time.time()
print('full: ',t2-t1)
