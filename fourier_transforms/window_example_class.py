import numpy as np

x=np.arange(1000)
x=x-x.mean()
y=np.random.randn(len(x))
yy=y+0.5*x

plt.figure(1)
plt.clf()
plt.plot(x,y)
plt.plot(x,yy)
plt.show()

plt.figure(2)
yft=np.fft.rfft(y)
yyft=np.fft.rfft(yy)
plt.clf()
plt.plot(np.abs(yft))
plt.plot(np.abs(yyft))
plt.show()

xx=np.pi/2*x/x.max()
win=np.cos(xx)
mywinft=np.fft.rfft(win*yy)
plt.plot(np.abs(mywinft))
plt.semilogy()
