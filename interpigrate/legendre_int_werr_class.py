import numpy as np
from matplotlib import pyplot as plt
plt.ion()

def lorentz(x):
    return 1/(1+x**2)

def rt(x):
    return 1/np.sqrt(x)

def get_weights(n):
    x=np.linspace(-1,1,n+1)
    P=np.zeros([n+1,n+1])
    P[:,0]=1
    P[:,1]=x
    for nn in range(1,n):  #evaluate the recurrence relation to get P_n
        P[:,nn+1]=((2*nn+1)*x*P[:,nn]-nn*P[:,nn-1])/(nn+1)
    Pinv=np.linalg.inv(P)
    wt=Pinv[0,:]
    return wt/wt.sum()*n


def integrate_leg_worker(f,a,b,n_group,ord):
    wts=get_weights(ord)
    npt=n_group*ord+1
    x=np.linspace(a,b,npt)
    y=f(x)
    dx=x[1]-x[0]
    tot=0
    for i in range(n_group):
        i_start=i*ord
        tot=tot+np.sum(wts*y[i_start:i_start+ord+1])
    return tot*dx

    
def integrate_leg(f,a,b,dx_targ=0.1,ord=4):
    #first, make sure we have consistent number of points given our order
    npt_rough=(b-a)/dx_targ
    n_group=int(np.round(npt_rough/ord)) #get the approximate number of order-sized blocks
    return integrate_leg_worker(f,a,b,n_group,ord)

def integrate_leg_werr(f,a,b,dx_targ=0.1,ord=4):
    #first, make sure we have consistent number of points given our order
    npt_rough=(b-a)/dx_targ
    n_group=int(np.round(npt_rough/ord)) #get the approximate number of order-sized blocks
    #make sure we have an even number of groups
    if n_group%2==1:
        n_group=n_group+1
    ans_fine=integrate_leg_worker(f,a,b,n_group,ord)
    ans_coarse=integrate_leg_worker(f,a,b,n_group//2,ord)
    return ans_fine,np.abs(ans_fine-ans_coarse)



#npt=n_group*ord+1 #make sure we now have exactly correct number of points
#    wts=get_weights(ord)
#    x=np.linspace(a,b,npt) #get my x values
#    dx=x[1]-x[0] #slightly sloppy version of dx
#    tot=0
#    y=f(x)
#    #sum weights times function values over each ord*dx interval
#    for i in range(n_group):
#        i_start=i*ord
#        tot=tot+np.sum(wts*y[i_start:i_start+ord+1])
#    return tot*dx
        

#print('second order weights are: ',get_weights(4))
#ord=8
#print('for order ',2,' weights are ',get_weights(ord)*ord)
#print('weight sum: ',np.sum(get_weights(ord)))

b=1.0
a=0.0
dx=0.2
val=integrate_leg(np.exp,a,b,dx,6)
ans=np.exp(b)-np.exp(a)
print('itegral is: ',val,' with error: ',val-ans)

a=-1.0
b=1.0
dx=0.2
val=integrate_leg(lorentz,a,b,dx,6)
ans=np.arctan(b)-np.arctan(a)
print('lorentz: ',val,' with error: ',val-ans)

val,err=integrate_leg_werr(lorentz,a,b,dx/2,4)
print('error estimate was: ',np.abs(err))
