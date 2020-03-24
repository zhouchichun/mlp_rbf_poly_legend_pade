import numpy as np 
#import matplotlib.pyplot as plt
import sys
import random

class give_batch():
    def __init__(self, x_range):
        self.x_range = x_range
        self.eps=0.00001
    def inter(self, batchsize):
        ret = np.linspace(self.x_range[0]+self.eps, self.x_range[1]-self.eps, batchsize)
        ret=np.reshape(ret,[batchsize,1])
        return ret
    def inter_random(self, batchsize):
        ret = np.random.uniform(self.x_range[0], self.x_range[1], batchsize)
        ret=np.reshape(ret,[batchsize,1])
        return ret

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    pi=3.141592653
    D=give_batch([-pi/2,pi/2])
    data=D.inter(100)
    print(data[:2])
    print(data[-2:])
    print(pi/2)
    plt.plot(D.inter(100))
    plt.show()
    