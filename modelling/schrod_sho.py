import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import eigh_tridiagonal
plt.ion()

#time-independent schrodinger is (-hbar^2/2m del^2 +V) Psi = E Psi
#for simple harmonic oscillator, V=kx^2/2
#if h bar = m = k = 1, then we have
#-1/2 del^2 Psi + x^2/2 Psi = E Psi
#and energies/corresponding wave functions are the eigenvalues/eigenvectors of the left
#grad is (Psi(x+dx)-Psi(x))/dx, and second deriv is
#grad of grad, or ((Psi(x+dx)-Psi(x))/dx - (Psi(x)-Psi(x-dx))/dx)/dx
#= (Psi(x+dx)-2 Psi(x) + Psi(x-dx))/dx^2

x=np.linspace(-10,10,2001)
V=0.5*x**2
dx=x[1]-x[0]
d=1/dx**2+V
e=-0.5/dx**2+0*x[:-1] #off-diagonal is one shorter
ee,vv=eigh_tridiagonal(d,e)
print('energies: ',ee[:5])
