import numpy as np
class one_devide_x():
    def forward(self,x):
        self.x=x
        z=1/self.x
        return z
    def backward(self,gradz):
        self.gradz=gradz
        gradx=self.gradz*(-1/self.x**2)
        return gradx 
class mat_plus():
    def __init__(self,b):
        self.b=b
    def forward(self,x):
        y=x+self.b
        return y
    def backward(self,grady):
        gradx=grady
        gradb=grady
        return gradb,gradx
class exp():
    def forward(self,x):
        self.x=x
        z=np.exp(x)
        return z
    def backward(self,gradz):
        self.gradz=gradz
        gradx=self.gradz*np.exp(self.x)
        return gradx
class multiply_1G():
    def __init__(self,multiplying):
        self.multiplying=multiplying
    def forward(self,x):
        self.x=x
        z=self.x*self.multiplying
        return z
    def backward(self,gradz):
        self.gradz=gradz
        gradx=self.gradz*self.multiplying
        return gradx
class plus_0G():
    def forward(self,x,y):
        self.x=x
        self.y=y
        z=self.x+self.y
        return z
    def backward(self,gradz):
        self.gradz=gradz
        gradx=self.gradz
        grady=self.gradz
        return gradx,grady
class multiply_0G():
    def forward(self,x,y):
        self.x=x
        self.y=y
        z=self.x*self.y
        return z
    def backward(self,gradz):
        self.gradz=gradz
        gradx=self.gradz*self.y
        grady=self.gradz*self.x
        return gradx,grady
class mat_multiply():
    def __init__(self,w):
        self.w=w
        self.wheight,self.wwidth=w.shape
    def forward(self,x):
        def get_execution():
            multiply_executions=np.empty([self.wheight,self.xwidth,self.wwidth],dtype=object)
            plus_executions=np.empty([self.wheight,self.xwidth,self.wwidth-1],dtype=object)
            for m in range(self.wheight):
                for n in range(self.xwidth):
                    for o in range(self.wwidth):
                        multiply_executions[m,n,o]=multiply_0G()
            for m in range(self.wheight):
                for n in range(self.xwidth):
                    for o in range(self.wwidth-1):
                        plus_executions[m,n,o]=plus_0G()
            self.multiply_executions=multiply_executions
            self.plus_executions=plus_executions
        def multiply_section():
            multiplied=np.zeros([self.wheight,self.xwidth,self.wwidth])
            for m in range(self.wheight):
                for n in range(self.xwidth):
                    for o in range(self.wwidth):
                        multiply_execution=self.multiply_executions[m,n,o]
                        multiplied[m,n,o]=multiply_execution.forward(self.w[m,o],self.x[o,n])
            return multiplied
        def plus_section():
            final=np.zeros([self.wheight,self.xwidth])
            for m in range(self.wheight):
                for n in range(self.xwidth):
                    plusx=multiplied[m,n,0]
                    for o in range(self.wwidth-1):
                        plusy=multiplied[m,n,o+1]
                        plus_execution=self.plus_executions[m,n,o]
                        plusz=plus_execution.forward(plusx,plusy)
                        plusx=plusz
                final[m,n]=plusx
            return final
        self.x=x
        self.xheight,self.xwidth=x.shape
        get_execution()
        multiplied=multiply_section()
        final=plus_section()
        return final
    def backward(self,gradz):
        self.gradz=gradz
        def plus_section():
            back_plused_grad=np.zeros([self.wheight,self.xwidth,self.wwidth])
            for m in range(self.wheight):
                for n in range(self.xwidth):
                    plusgradz=self.gradz[m,n]
                    for o in range(self.wwidth-2,-1,-1):
                        plus_execution=self.plus_executions[m,n,o]
                        plusgradx,plusgrady=plus_execution.backward(plusgradz)
                        back_plused_grad[m,n,self.wwidth-1-o]=plusgrady
                    back_plused_grad[m,n,0]=plusgradx
            return back_plused_grad
        def multiply_section():
            gradw=np.zeros([self.wheight,self.wwidth])
            gradx=np.zeros([self.xheight,self.xwidth])
            for m in range(self.wheight):
                for n in range(self.xwidth):
                    for o in range(self.wwidth):
                        multiply_execution=self.multiply_executions[m,n,o]
                        gradw[m,o]=multiply_execution.backward(back_plused_grad[m,n,o])[0]
                        gradx[o,n]+=multiply_execution.backward(back_plused_grad[m,n,o])[1]
            return gradw,gradx
        back_plused_grad=plus_section()
        gradw,gradx=multiply_section()
        return gradw,gradx




