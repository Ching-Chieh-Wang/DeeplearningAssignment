import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from tqdm import tqdm
from Activation import ReLU,Softmax
from Layers import Dense,Input
from Perceptron import MLP
from skimage import feature
from lossfunc import CrossEntropyLoss
def loaddata(trainpathtxt,valpathtxt,testpathtxt,size):
    def try_get_img_imfo():
        img_paths=open(trainpathtxt)
        img=cv.imread(img_paths.readline().split(' ')[0])
        img=cv.resize(img,size)
        fd = feature.hog(img, orientations=9,pixels_per_cell=(16, 16),cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2-Hys')
        img_paths.close()
        return fd.shape
    def count_imgs(txtlines):
        for m,line in enumerate(txtlines):
            continue
        txtlines.seek(0)
        return (m+1)
    def loading(txtpath):
        txtlines=open(txtpath)
        imgs_count=count_imgs(txtlines)
        datas=np.zeros((imgs_count,*xshape,1),dtype=float)
        labels=np.zeros(imgs_count)
        for n,line in tqdm(enumerate(txtlines),total=imgs_count):
            img_path=line.split(' ')[0]
            img=cv.imread(img_path)
            img=cv.resize(img,size)
            fd = np.expand_dims(feature.hog(img, orientations=9, pixels_per_cell=(16, 16),cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2-Hys'),axis=-1)
            datas[n]=fd
            label=line.split(' ')[1]
            labels[n]=label
        txtlines.close()
        
        return datas,labels
    xshape=try_get_img_imfo()
    print('/nloading train datum')
    xtrain,ytrain=loading(trainpathtxt)
    print('/nloading val datum')
    xval,yval=loading(valpathtxt)
    print('/nloading test datum')
    xtest,ytest=loading(testpathtxt)
    return xtrain,ytrain,xval,yval,xtest,ytest
def onehotencoding(y,classes):
    onehot_y=np.zeros([y.shape[0],classes,1],dtype=np.bool)
    for n,i in enumerate(y):
        onehoted=np.zeros([classes,1],dtype=np.bool)
        onehoted[int(i),0]=1
        onehot_y[n]=onehoted
    return onehot_y
    
    




trainpathtxt='train.txt'
valpathtxt='val.txt'
testpathtxt='test.txt'
'''
此作業使用HOG方法獲取feature，如需調整參數須至loaddata進行調整
'''
xtrain,ytrain,xval,yval,xtest,ytest=loaddata(trainpathtxt,valpathtxt,testpathtxt,size=(128,128))
'''
onehotencodeing(data,classes)
將資料轉為one hot 模式
'''
ytrain=onehotencoding(ytrain,50)
yval=onehotencoding(yval,50)
ytest=onehotencoding(ytest,50)

'''
model.add 可以增加layers
第一層一定需使用Input層
Input(feature size,output size,activation=activation)
Dense(output size)
'''
model=MLP()
model.add(Input(1764,100,activation=ReLU))
model.add(Dense(50,activation=Softmax))
model.compile(CrossEntropyLoss,lr=0.01)
model.fit(xtrain,ytrain,xval,yval,epochs=15)
model.plot_result()










