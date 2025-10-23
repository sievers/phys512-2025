import numpy as np

class Complex2:
    def __init__(self,r,i):
        self.r=r
        self.i=i
    def abs(self):
        return np.sqrt(self.r**2+self.i**2)
def add(self,a):
    if isinstance(a,Complex): #this is true if a is a complex class instance
        return Complex(self.r+a.r,self.i+a.i)
    else:
        return Complex(self.r+a,self.i)



class Complex:
    def __init__(self,r,i):
        self.r=r
        self.i=i
    def abs(self):
        return np.sqrt(self.r**2+self.i**2)
    def add(self,a):
        if isinstance(a,Complex): #this is true if a is a complex class instance
            return Complex(self.r+a.r,self.i+a.i)
        else:
            return Complex(self.r+a,self.i)

    def __repr__(self):
        if self.i>0:
            return repr(self.r)+' + '+repr(self.i)+'J'
        else:
            return repr(self.r)+' - '+repr(-self.i)+'J'
    def __add__(self,a):
        return self.add(a)
    def __radd__(self,a):
        return self.add(a)
    def copy(self):
        return Complex(self.r,self.i)
    def __mul__(self,a):    
        if isinstance(a,Complex):
            return Complex(a.r*self.r-a.i*self.i,self.r*a.i+self.i*a.r)
        else:
            return Complex(self.r*a,self.i*a)
        
a=Complex(2,3)
#a=2.3
b=Complex(3,-4)
#b=4
#c=a.add(b)
c=a+b
c=b+a
#print("abs of a is ",a.abs())
print("c is ",c)
