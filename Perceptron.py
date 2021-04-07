import numpy as np
from lossfunc import CrossEntropyLoss
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
def init(layers): 
    for n,layer in enumerate(layers):
        if type(layer).__name__=='Input':
            layer.init()
        else:
            input_neurons=layers[n-1].output_neurons
            layer.init(input_neurons)

class MLP():
    def __init__(self):
        self.layers=[]
    def add(self,layer,**args):
        self.layers.append(layer)
    def compile(self,lossfunc,lr):
        self.lossfunc=lossfunc
        self.lr=lr
    def predict(self,X,Y):
        correct=0
        for x,y in tqdm(zip(X,Y),total=X.shape[0]):
            for layer in self.layers:
                ypred=layer.forward(x)
                x=ypred
            loss=self.lossfunc.loss(y,ypred)
            if np.argmax(y)==np.argmax(ypred):
                correct+=1
        accuracy=correct/X.shape[0]
        return loss,accuracy
    def fit(self,X,Y,Xval,Yval,epochs):
        '''
        init would init weights and create computation environment at each layer
        '''
        init(self.layers)
        self.val_losses,self.val_accs=[],[]
        self.losses,self.accs=[],[]
        for epoch in range(epochs): # turn x into multiple batches
            np.random.seed(47)
            np.random.shuffle(X)
            np.random.seed(47)
            np.random.shuffle(Y)
            total_datum=X.shape[0]
            for n,(x,y) in tqdm(enumerate(zip(X,Y)),ascii=True,desc='epoch'+str(epoch),total=total_datum):
                for layer in self.layers:
                    ypred=layer.forward(x)
                    x=ypred
                loss=self.lossfunc.loss(y,ypred)
                grady=self.lossfunc.gradient(y,ypred)
                for layer in reversed(self.layers):
                    gradx=layer.backward(grady,self.lr)
                    grady=gradx
            loss,acc=self.predict(X,Y)
            val_loss,val_acc=self.predict(Xval,Yval)
            self.losses.append(loss)
            self.accs.append(acc)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            print()
            print('epoch{} ,loss={} ,acc={} , val_loss={},val_acc={}'.format(epoch+1,loss,acc,val_loss,val_acc))
            print()
    def plot_result(self):
        plt.plot(self.losses,label='train loss')
        plt.plot(self.val_losses,label='val loss')
        plt.legend()
        plt.title('loss')
        plt.show()
        plt.plot(self.accs,label='train accuracy')
        plt.plot(self.val_accs,label='val accuracy')
        plt.legend()
        plt.show()
        

    
    
    

