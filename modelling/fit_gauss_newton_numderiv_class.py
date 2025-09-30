import numpy as np
from matplotlib import pyplot as plt
plt.ion()

def gauss(m,x):
    amp=m[0]
    x0=m[1]
    vinv=m[2]
    return np.exp(-0.5*(x-x0)**2*vinv)*amp

def get_A(m,x,fun=gauss,dm=None):
    if dm is None:
        dm=0.01*np.abs(m)
    A=np.zeros([len(x),len(m)])
    for i in range(len(m)):
        mm=m.copy()
        mm[i]=mm[i]+dm[i] #evaluate one parameter shifted positive
        f_plus=fun(mm,x)
        mm[i]=m[i]-dm[i]  #shift negative
        f_minus=fun(mm,x)
        A[:,i]=(f_plus-f_minus)/(2*dm[i])
    return A

def get_A_analytic(m,x):
    amp=m[0]
    x0=m[1]
    vinv=m[2]
    #make matrix of derivatives
    A=np.zeros([len(x),len(m)])
    #deriv of A w.r.t amp
    A[:,0]=np.exp(-0.5*(x-x0)**2*vinv)
    A[:,1]=-gauss(m,x)*(-(x-x0)*vinv)
    A[:,2]=gauss(m,x)*(-0.5*(x-x0)**2)
    return A
m_true=[1.0,0.0,1.0] #unit gaussian

x=np.linspace(-5,5,1001)
y_true=gauss(m_true,x)
y=y_true+np.random.randn(len(x))

plt.clf()
plt.plot(x,y_true)
plt.plot(x,y,'.')
plt.show()

m_guess=np.asarray(m_true)
dm=np.ones(3)*1e-3 #hardware to 10^-3 for parameter steps

niter=10
m_cur=m_guess
for iter in range(niter):
    A=get_A(m_cur,x,gauss,dm)
    mod=gauss(m_cur,x)
    r=y-mod #difference between data and model
    lhs=2*A.T@A
    rhs=-2*A.T@r
    step=-np.linalg.inv(lhs)@rhs
    m_cur=m_cur+step
    print('current params: ',m_cur)
    print('current step: ',step)
y_fit=gauss(m_cur,x)
plt.plot(x,y_fit,'r')
