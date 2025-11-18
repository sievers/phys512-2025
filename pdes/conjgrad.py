import numpy as np


def conjgrad(A,b,x=0,niter=100):
    try:
        r=b-A@x
    except:
        r=b
    p=r.copy()
    for k in range(niter):
        Ap=A@p #this should take almost all the time
        rtr=np.sum(r*r) #r@r
        pAp=np.sum(p*Ap) #p@Ap
        alpha=rtr/pAp
        x=x+alpha*p
        r=r-alpha*Ap
        rtr_new= np.sum(r*r) #r@r
        beta=rtr_new/rtr
        p=r+beta*p
        print('rtr on step ',k,' is ',rtr_new)
    return x
        
