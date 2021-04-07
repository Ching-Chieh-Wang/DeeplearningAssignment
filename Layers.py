import numpy as np
import Computation_graph as cg

class Dense():
    def __init__(self,output_neurons,activation):
        self.output_neurons=output_neurons
        self.activation=activation
    def init(self,input_neurons):
        self.input_neurons=input_neurons
        limit   = 1 / np.sqrt(input_neurons)
        self.w=np.random.uniform(-limit,limit,(self.output_neurons,self.input_neurons))
        self.b=np.zeros([self.output_neurons,1])
        self.executions=[cg.mat_multiply(self.w),cg.mat_plus(self.b),self.activation()]
    def forward(self,x):
        for execution in self.executions:
            z=execution.forward(x)
            x=z
        return z
    def backward(self,gradz,lr):
        for execution in reversed(self.executions):
            grady,gradx=execution.backward(gradz)
            gradz=gradx
            if type(execution).__name__=='mat_plus':
                self.b-=lr*grady
            if type(execution).__name__=='mat_multiply':
                self.w-=lr*grady
        return gradx
class Input(Dense):
    def __init__(self,input_neurons,output_neurons,activation):
        self.input_neurons=input_neurons
        self.output_neurons=output_neurons
        self.activation=activation
    def init(self):
        limit   = 1 / np.sqrt(self.input_neurons)
        self.w=np.random.uniform(-limit,limit,(self.output_neurons,self.input_neurons))
        self.b=np.ones([self.output_neurons,1])
        self.executions=[cg.mat_multiply(self.w),cg.mat_plus(self.b),self.activation()]
    def forward(self,x):
        z=super().forward(x)
        return z
    def backward(self,gradz,lr):
        super().backward(gradz,lr)


        