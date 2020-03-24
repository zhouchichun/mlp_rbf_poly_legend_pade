import numpy as np
pi=3.141592653
a=-1*pi/2
A=0

b=pi/2
B=2.0

#g=10.0
exact=0.993459
def give_target(x):
    x=np.reshape(x,[len(x)])
    y=np.sin(2*x**2)*np.exp(-(x-1)**2)+x*(2.7/pi)+1.35
    y=np.reshape(y,[-1,1])
    return y

if __name__=="__main__":
    import matplotlib.pyplot as plt 
    x=np.linspace(a,b,100)
    print(give_target(np.array([a])))
    print(give_target(np.array([b])))
    y=give_target(x)
    plt.plot(y)
    plt.show()
