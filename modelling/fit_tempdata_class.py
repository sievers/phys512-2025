import numpy as np
from openpyxl import load_workbook
from matplotlib import pyplot as plt
plt.ion()
plt.figure(1)
crud=load_workbook('A02_GAIN_MIN20_373.xlsx')
sheet=crud['All']

nu=[sheet['A'+repr(i+1)].value for i in range(4096)]
nu=np.asarray(nu)
gain=[sheet['B'+repr(i+1)].value for i in range(4096)]
gain=np.asarray(gain)
nu=nu/1e6 #convert frequency from Hz to MHz

nu_min=15

ii=nu>nu_min
nu=nu[ii]
gain=gain[ii]
plt.clf()
plt.plot(nu,gain)
plt.show()

xx=nu-nu.min()
xx=xx/xx.max() #now on 0 to 1
xx=2*xx-1  #now on -1 to 1
print(xx.min(),xx.max()) #should be -1,1

order=8
A=np.polynomial.legendre.legvander(xx,order)
lhs=A.T@A
rhs=A.T@gain
fitp=np.linalg.inv(lhs)@rhs
#<d> = Am
pred=A@fitp
plt.plot(nu,pred)
print('error squared: ',np.mean((gain-pred)**2))
plt.figure(2)
plt.clf()
plt.plot(nu,gain-pred)
plt.show()
r=gain-pred #the residual in our fit
N=np.mean(r**2)
print('Noise variance/sigma are: ',N,np.sqrt(N))
#noise is N(scalar)*I
par_covar=np.linalg.inv(A.T@A/N)
par_errs=np.sqrt(np.diag(par_covar))
print('parameter errors: ',par_errs)
model_covar=A@par_covar@A.T
model_sigma=np.sqrt(np.diag(model_covar))

