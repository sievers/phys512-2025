import numpy as np
from matplotlib import pyplot as plt
import time


#we could use either of the following sigmoid functions
#for our purposes, they're pretty interchangeable

#def sigmoid(z):
#    return 1.0/(1.0+np.exp(-z))
#def sigmoid_prime(z):
#    """Derivative of the sigmoid function."""
#    return sigmoid(z)*(1-sigmoid(z))

def sigmoid(z):
    return 0.5+np.arctan(z)/np.pi

def sigmoid_inv(z):
    return np.tan(np.pi*(z-0.5))

def sigmoid_prime(z):
    return 1.0/np.pi/(1+z**2)


class Net:
    #OK, so python lets you add vector to matrix as long as last axis matches
    #so arrays of input have to be size [n,784]
    #That means we'll have xW+b, where W[0] is [784,30] e.g.
        
    def __init__(self,sizes,weights=None,bias=None):
        self.sizes=[sizes[i] for i in range(len(sizes))]
        self.depth=len(self.sizes)-1
        if weights is None:
            self.weights=[None]*self.depth
            for i in range(self.depth):
                self.weights[i]=np.random.randn(self.sizes[i],self.sizes[i+1])
        else:
            self.weights=weights
            for i in range(self.depth):
                self.weights[i]=np.squeeze(self.weights[i])
        if bias is None:
            self.bias=[None]*self.depth
            for i in range(self.depth):
                self.bias[i]=np.random.randn(self.sizes[i+1])
        else:
            self.bias=bias
            for i in range(self.depth):
                self.bias[i]=np.squeeze(self.bias[i])
    def eval(self,x_in):
        #evaluate our network for input signals.
        #code is set up such that we can do many input at the same time
        #we return intermediate products as well that are useful for
        #backpropagation
        x=[None]*self.depth
        x[0]=x_in@self.weights[0]+self.bias[0]
        for i in range(1,self.depth):
            x[i]=sigmoid(x[i-1])@self.weights[i]+self.bias[i]
        return sigmoid(x[-1]),x
    def backprop(self,x_in,y):
        #calculate the gradient of the cost function (assumed quadratic here)
        #given a set of input data (x_in) and target values (y).
        #this gives us the derivative of the cost function with respect to each
        #parameter in the network.  This supports many simultaneous inputs (i.e. x_in
        #and y can be 2D arrays), which is much, much faster than looping over inputs.

        pred,x=self.eval(x_in) #we'll need the outputs of all the layers
        wgrad=[np.zeros_like(self.weights[i]) for i in range(self.depth)]
        bgrad=[np.zeros_like(self.bias[i]) for i in range(self.depth)]
        xgrad=[np.zeros_like(x[i]) for i in range(self.depth)]
        #cost is sum( (y-sigmoid(Wx+b))^2).  let r=y-sigmoid(),
        #then gradient is -2r d(sigmoid)/dp for parameter p
        #and d(sigmoid)/dp is sigmoid_prime(Wx+b)*d(Wx+b)/dp
        #we're doing the last layer here since it's slightly different
        #than the other layers because the final layer sees the cost
        #function directly.
        #NB - other cost functions are trivial to use, you just need
        #to update the cost function derivative.  d(C(f))/dp=C' df/dp
        #Most of the work is handling the internal derivatives, so you could
        #change cost functions just by swapping them out in C'
        r=y-sigmoid(x[-1]) 
        vec=-2*r*sigmoid_prime(x[-1])
        wgrad[-1]=sigmoid(x[-2]).T@vec
        bgrad[-1]=np.sum(vec,axis=0)
        xgrad[-1]=self.weights[-1]@(vec.T)
        x.append(x_in) #add the starting values at the end for ease of indexing

        #now that we have the gradient of the final layer, we can do the preceeding layers
        #by working out how the final layer depends on the previous layer.  Once you know
        #that, you use the chain rule to get how the final cost function depends on the
        #previous layer.  You can keep doing this all the way through your network layers
        #in exactly the same way
        for i in range(self.depth-2,-1,-1):
            tmp=sigmoid_prime(x[i])*(xgrad[i+1].T)
            bgrad[i]=np.sum(tmp,axis=0)
            wgrad[i]=x[i-1].T@tmp
            xgrad[i]=(tmp@(self.weights[i].T)).T
        return wgrad,bgrad,xgrad

    def SGD(self,niter,batch_size,dat,ans,rate,vdat=None,vans=None):
        #do gradient descent.  Due to laziness, this isn't very stochastic, but
        #it also doesn't matter very much.
        n=dat.shape[0]
        nbatch=n//batch_size
        fracs=np.zeros(niter)
        for iter in range(niter):
            #we do each iteration by looping over our mini-batches
            for batch in range(nbatch):
                i1=batch_size*batch
                i2=i1+batch_size
                vv=np.squeeze(dat[i1:i2,:])
                dd=np.squeeze(ans[i1:i2,:])
                if i2>n:
                    i2=n
                #get the gradients using our current minibatch
                wgrad,bgrad,agrad=self.backprop(vv,dd)

                #now that we have gradients, update our network parameters
                for j in range(self.depth):                
                    self.weights[j]=self.weights[j]-rate/n*wgrad[j]
                    self.bias[j]=self.bias[j]-rate/n*bgrad[j]
            #we can report what fraction of our validation data are
            #classified correctly to see how our fit is working
            if not(vdat is None):
                stuff,crap=self.eval(vdat)
                frac=np.mean(np.argmax(stuff,axis=1)==vans)
                fracs[iter]=frac
                print('iter ',iter,', correct fraction: ',frac)
        return fracs #so we can plot how well we trained


plt.ion()

stuff=np.load('mnist.npz')
td0=stuff['td0']/256.0
td1=stuff['td1']
vd0=stuff['vd0']/256.0
vd1=stuff['vd1']
tst0=stuff['tst0']/256.0
tst1=stuff['tst1']
wt0=np.load('weights0.npy')
wt1=np.load('weights1.npy')
b0=np.load('biases0.npy')
b1=np.load('biases1.npy')

sizes=[wt0.shape[1],wt0.shape[0],wt1.shape[0]]

net=Net(sizes)
fracs=net.SGD(500,100,td0,td1,1000.0,vd0,vd1)
