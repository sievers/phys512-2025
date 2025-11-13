import numpy as np


def conjgrad(A,b,x=0,niter=100):
    try:
        r=b-A@x
    except:
        r=b
    p=r.copy()
    for k in range(niter):
        Ap=A@p #this should take almost all the time
        rtr=r@r
        pAp=p@Ap
        alpha=rtr/pAp
        x=x+alpha*p
        r=r-alpha*Ap
        rtr_new=r@r
        beta=rtr_new/rtr
        p=r+beta*p
        print('rtr on step ',k,' is ',rtr_new)
    return x
        

N=5000
A=np.random.randn(N,N)
A=A.T@A+np.eye(N)*10
e,v=np.linalg.eigh(A)
b=np.random.randn(N)
print('condition number is ',e.max()/e.min())
x=conjgrad(A,b)
r=b-A@x
print("residual rms is ",np.std(r))
x_true=np.linalg.inv(A)@b
print('scatter between CG and true: ',np.std(x-x_true))
print('scatter of truth: ',np.std(x_true))
