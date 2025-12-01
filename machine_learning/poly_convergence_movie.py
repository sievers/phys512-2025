#We'll generate data from a random polynomial, then fit
#polynomials of varying order to half the data.   As we
#increase the order of the poly, chi^2 will drop happily
#as the order goes up.  We check against chi^2 for the
#other half of the data.  At first that goes down, but as fit
#order goes past order actually in the data, then chi^2 goes
#up.  We can get an idea of what order to use by watching
#for that turnup.

import numpy as np
from matplotlib import pyplot as plt
plt.ion()


x=np.linspace(-1,1,201)
ord=10 #true order of our data
coeffs=np.random.randn(ord+1) #make random coefficiens
mat=np.polynomial.chebyshev.chebvander(x,ord) 
y_true=mat@coeffs #and apply them to chebyshev polynomials
plt.clf();plt.plot(x,y_true);plt.show()
N=0.5
y=y_true+N*np.random.randn(len(x))
plt.plot(x,y,'.')

max_ord=60

mat=np.polynomial.legendre.legvander(x,max_ord)
chi=np.zeros(max_ord) #keep track of chi^2 for the data we fit to
chi_others=np.zeros(max_ord) #and evaluate it for the data we didn't fit
ymax=y.max()
ymin=y.min()
dy=ymax-ymin
yspan=[ymin-0.1*dy,ymax+0.1*dy]
for i in range(1,max_ord):
    mm=mat[:,:i] #pick out polynomials up to current order
    mm1=mm[::2,:] #and pick out even/odd data parts of current polynomial matrix
    mm2=mm[1::2,:]
    lhs=mm1.T@mm1  #do a least-squares fit
    rhs=mm1.T@y[::2]
    fitp=np.linalg.inv(lhs)@rhs
    pred=mm@fitp
    #pred1=mm1@fitp #evaluate fit for even/odd points
    #pred2=mm2@fitp
    pred1=pred[::2]
    pred2=pred[1::2]
    chi[i]=np.sum((y[::2]-pred1)**2) #find chi^2 for the even points (the ones we used)
    chi_others[i]=np.sum((y[1::2]-pred2)**2) #and for the odd points
    plt.clf()
    plt.plot(x[::2],y[::2],'.')
    plt.plot(x[1::2],y[1::2],'r.')
    plt.plot(x,pred)
    plt.legend(['RMS: '+repr(np.sqrt(chi[i]*2/len(x))),'RMS: '+repr(np.sqrt(chi_others[i]*2/len(x)))],loc='upper right')
    plt.title('Poly order: '+repr(i))
    plt.ylim(yspan)
    plt.pause(0.5)
    
#plt.clf()
#plt.plot(chi)
#plt.plot(chi_others)
#plt.show()
#chi_pred=N**2*(len(x)/2-np.arange(max_ord))
#plt.plot(chi_pred[1:],'r')
