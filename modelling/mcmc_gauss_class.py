import numpy as np
from matplotlib import pyplot as plt
plt.ion()


def gauss(p,x):
    amp=p[0]
    x0=p[1]
    sig=p[2]
    return np.exp(-0.5*(x-x0)**2/sig**2)*amp

def get_chi(p,f,x,d,N):
    y=f(p,x)
    r=y-d
    return np.sum(r**2/N)

def mcmc(p,f,x,d,N,dpar,nsamp=150000):
    #f(x,p) should be our predicted data
    npar=len(p)
    chain=np.zeros([nsamp,npar]) #allocate space for the chain
    chi_cur=get_chi(p,f,x,d,N)
    chain[0,:]=p
    for i in range(1,nsamp):
        p_trial=p+np.random.randn(npar)*dpar
        chi_trial=get_chi(p_trial,f,x,d,N)
        prob_ratio=np.exp(-0.5*(chi_trial-chi_cur))
        accept=np.random.rand()<prob_ratio
        if (accept):
            p=p_trial
            chi_cur=chi_trial #I forgot this line in class.  this updates the current chi^2
        chain[i,:]=p
    return chain

x=np.linspace(-2,2,1001)
pars=np.asarray([1,0,1])
y_true=gauss(pars,x)
y=y_true+np.random.randn(len(x))*0.2
p_guess=0.5*pars #start somewhat off
dp=0.05*np.ones(3) #step size hardwired in

chain=mcmc(p_guess,gauss,x,y,0.04,dp)
