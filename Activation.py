import numpy as np
class ReLU():
    def __init__(self):
        pass
    def forward(self,x):
        self.x=x
        y=np.where(self.x>=0,self.x,0)
        return y
    def backward(self,grady):
        gradx=np.where(grady>=0,1,0)
        return None,gradx
class Softmax():    
    def forward(self,x):
        self.x=x
        e_x = np.exp(self.x )
        self.y=e_x / np.sum(e_x)
        return self.y
    def backward(self,grady):
        gradx=np.sum(grady*-self.y)*self.y+self.y*grady
        return None,gradx

