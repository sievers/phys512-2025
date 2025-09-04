import numpy as np

#we have f(x=-1)=yl, f(x=0)=y0, f(x=1)=yr
#we want quadratic ax^2 + bx +c that goes through these points
#c=y0
# yl = a -b +c
# yr = a +b +c
# y0 = c

#yl+yr=2a+2c, yl-2y0+yr=2a, a=(yl+yr-2y0)/2
#area is integral from -1 to 1, integral is cx + ax^3/3 | -1,1
#gives 2c + 2/3a = 2y0 + 2/3 (yl +yr - 2y0)/2 = 1/3 (6y0 + yl +yr -2y0)
#= 1/3(yl+4y0+yr)
#a bunch of y values: first interval is 1/3 (y[0]+4y[1]+y[2]) +
# + 1/3(y[2]+4y[3]+y[4]) + 1/3 (y[4]+4y[5]+y[6])+...
#1/3 (y[0]+4y[1] +2y[2]+4y[3]+2y[4]+...+y[-1])

def f(x):
    return 1/x
def simpson_raw(x,y):
    dx=x[1]-x[0]
    return dx*(y[0]+y[-1]+4*np.sum(y[1:-1:2])+2*np.sum(y[2:-2:2]))/3

def simpson(f,a,b,npt):
    x=np.linspace(a,b,npt)
    y=f(x)
    return simpson_raw(x,y)
a=1
b=2
print('log from simpson is ',simpson(f,a,b,41),np.log(b)-np.log(a))
