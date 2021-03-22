import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from skimage import feature
from tqdm import tqdm
import random
import argparse

class Softmax:

    __slots__ = ("epochs", "learningRate", "batchSize", "regStrength", "wt", "momentum", "velocity")
    def __init__(self, epochs, learningRate, batchSize, regStrength, momentum):

        self.epochs = epochs
        self.learningRate = learningRate
        self.batchSize = batchSize
        self.regStrength = regStrength
        self.momentum = momentum
        self.velocity = None
        self.wt = None

    def train(self, xtrain, ytrain, xval, yval):

        D = xtrain.shape[1]  # dimensionality
        label = np.unique(ytrain)
        numOfClasses = len(label) # number of classes
        ytrainEnc = self.oneHotEncoding(ytrain, numOfClasses)
        yvalEnc = self.oneHotEncoding(yval, numOfClasses)
        self.wt = 0.001 * np.random.rand(D, numOfClasses)
        self.velocity = np.zeros(self.wt.shape)
        trainLosses = []
        valLosses = []
        trainAcc = []
        valAcc = []
        for e in range(self.epochs): # loop over epochs
            trainLoss = self.SGDWithMomentum(xtrain, ytrainEnc)
            valLoss, dw = self.computeLoss(xval, yvalEnc)
            trainAcc.append(self.meanAccuracy(xtrain, ytrain))
            valAcc.append(self.meanAccuracy(xval, yval))
            trainLosses.append(trainLoss)
            valLosses.append(valLoss)
            print("{:d}\t->\ttrainL : {:.7f}\t|\tvalL : {:.7f}\t|\ttrainAcc : {:.7f}\t|\tvalAcc: {:.7f}"
                  .format(e, trainLoss, valLoss, trainAcc[-1], valAcc[-1]))
        return trainLosses, valLosses, trainAcc, valAcc

    def SGDWithMomentum(self, x, y):

        losses = []
        randomIndices = random.sample(range(x.shape[0]), x.shape[0])
        x = x[randomIndices]
        y = y[randomIndices]
        for i in range(0, x.shape[0], self.batchSize):
            Xbatch = x[i:i+self.batchSize]
            ybatch = y[i:i+self.batchSize]
            loss, dw = self.computeLoss(Xbatch, ybatch)
            self.velocity = (self.momentum * self.velocity) + (self.learningRate * dw)
            self.wt -= self.velocity
            losses.append(loss)
        return np.sum(losses) / len(losses)

    def softmaxEquation(self, scores):
        """
        It calculates a softmax probability
        :param scores: A matrix(wt * input sample)
        :return: softmax probability
        """
        scores -= np.max(scores)
        prob = (np.exp(scores).T / np.sum(np.exp(scores), axis=1)).T
        return prob

    def computeLoss(self, x, yMatrix):
        """
        It calculates a cross-entropy loss with regularization loss and gradient to update the weights.
        :param x: An input sample
        :param yMatrix: Label as one-hot encoding
        :return:
        """
        numOfSamples = x.shape[0]
        scores = np.dot(x, self.wt)
        prob = self.softmaxEquation(scores)

        loss = -np.log(np.max(prob)) * yMatrix
        regLoss = (1/2)*self.regStrength*np.sum(self.wt*self.wt)
        totalLoss = (np.sum(loss) / numOfSamples) + regLoss
        grad = ((-1 / numOfSamples) * np.dot(x.T, (yMatrix - prob))) + (self.regStrength * self.wt)
        return totalLoss, grad

    def meanAccuracy(self, x, y):
        """
        It calculates mean-per class accuracy
        :param x: Input sample
        :param y: label sample
        :return: mean-per class accuracy
        """
        predY = self.predict(x)
        predY = predY.reshape((-1, 1))  # convert to column vector
        return np.mean(np.equal(y, predY))

    def predict(self, x):
        """
        It predict the label based on input sample and a model
        :param x: Input sample
        :return: predicted label
        """
        return np.argmax(x.dot(self.wt), 1)
    def predict_top5(self, x):
        """
        It predict the label based on input sample and a model
        :param x: Input sample
        :return: predicted label
        """
        predict_labels=[]
        for i in x:
            predict_labels.append([np.argpartition(i.dot(self.wt), -5)[-5:]])
        return predict_labels
    def meanAccuracy_top5(self, x, y):
        predY = self.predict_top5(x)
        match=0
        for n,i in enumerate(predY):
            if i in y[n]:
                match+=1
        return match/y.shape[0]

    def oneHotEncoding(self, y, numOfClasses):
        """
        Convert a vector into one-hot encoding matrix where that particular column value is 1 and rest 0 for that row.
        :param y: Label vector
        :param numOfClasses: Number of unique labels
        :return: one-hot encoding matrix
        """
        y = np.asarray(y, dtype='int32')
        if len(y) > 1:
            y = y.reshape(-1)
        if not numOfClasses:
            numOfClasses = np.max(y) + 1
        yMatrix = np.zeros((len(y), numOfClasses))

        yMatrix[np.arange(len(y)), y] = 1
        return yMatrix


def plotGraph(trainLosses, valLosses, trainAcc, valAcc):

    plt.subplot(1, 2, 1)
    plt.plot(trainLosses, label="train loss")
    plt.plot(valLosses, label="val loss")
    plt.legend(loc='best')
    plt.title("Epochs vs. Cross Entropy Loss")
    plt.xlabel("Number of Iteration or Epochs")
    plt.ylabel("Cross Entropy Loss")

    plt.subplot(1, 2, 2)
    plt.plot(trainAcc, label="train Accuracy")
    plt.plot(valAcc, label="val Accuracy")
    plt.legend(loc='best')
    plt.title("Epochs vs. Mean per class Accuracy")
    plt.xlabel("Number of Iteration or Epochs")
    plt.ylabel("Mean per class Accuracy")
    plt.show()


def loadData(trainFilePath, valFilePath,testFilePath):
 
    xtrain = []
    ytrain = []
    xval = []
    yval = []
    xtest=[]
    ytest=[]
    train_path_txt=open(trainFilePath)
    # capture features using HOG from skimage.feature.hog
    for line in tqdm(train_path_txt,total=63325):
        img=cv.imread(line.split(' ')[0])
        img=cv.resize(img,(128,128))
        fd = feature.hog(img, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2-Hys')
        xtrain.append(fd)
        ytrain.append([int(line.split(' ')[1])])
    train_path_txt.close()
    val_path_txt=open(valFilePath)   
    for line in tqdm(val_path_txt,total=450):
        img=cv.imread(line.split(' ')[0])
        img=cv.resize(img,(128,128))
        fd = feature.hog(img, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2-Hys')
        xval.append(fd)
        yval.append([int(line.split(' ')[1])])
    val_path_txt.close()
    test_path_txt=open(testFilePath)  
    for line in tqdm(test_path_txt,total=450):
        img=cv.imread(line.split(' ')[0])
        img=cv.resize(img,(128,128))
        fd = feature.hog(img, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2-Hys')
        xtest.append(fd)
        ytest.append([int(line.split(' ')[1])])
    test_path_txt.close()
    return np.array(xtrain), np.array(ytrain), np.array(xval), np.array(yval),np.array(xtest),np.array(ytest)


if __name__ == "__main__":

    # txt file here
    xtrain, ytrain, xval, yval ,xtest,ytest = loadData('train.txt', 'val.txt','test.txt')
    # hyperparameter tuning here
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", dest="epochs", default=2500,
                        type=int, help="Number of epochs")
    parser.add_argument("-lr", "--learningrate", dest="learningRate", default=0.001,
                        type=float, help="Learning rate or step size")
    parser.add_argument("-bs", "--batchSize", dest="batchSize", default=200,
                        type=int, help="Number of sample in mini-batches")
    parser.add_argument("-r", "--regStrength", dest="regStrength", default=0.001,
                        type=float, help="L2 weight decay regularization lambda value")
    parser.add_argument("-m", "--momentum", dest="momentum", default=0.005,
                        type=float, help="A momentum value")

    args = parser.parse_args()

    print(
        "Epochs: {} | Learning Rate: {} | Batch Size: {} | Regularization Strength: {} | "
        "Momentum: {} |".format(
            args.epochs,
            args.learningRate,
            args.batchSize,
            args.regStrength,
            args.momentum
        ))

    epochs = int(args.epochs)
    learningRate = float(args.learningRate)
    batchSize = int(args.batchSize)
    regStrength = int(args.regStrength)
    momentum = int(args.momentum)
    # using perceptron class named Softmax
    sftmx = Softmax(epochs=epochs, learningRate=learningRate, batchSize=batchSize,
                       regStrength=regStrength, momentum=momentum)
    trainLosses, valLosses, trainAcc, valAcc = sftmx.train(xtrain, ytrain, xval, yval)
    plotGraph(trainLosses, valLosses, trainAcc, valAcc)
    # show top-1 and top-5 accuracy
    print('val top 1 acc=',sftmx.meanAccuracy(xval,yval))
    print('val top 5 acc=',sftmx.meanAccuracy_top5(xval,yval))
    print('test top 1 acc=',sftmx.meanAccuracy(xtest,ytest))
    print('test top 5 acc=',sftmx.meanAccuracy_top5(xtest,ytest))
    # classify using random forest and xgboost
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    rfc=RandomForestClassifier()
    rfc.fit(xtrain,np.ravel(ytrain))
    print('acc using randomforest classifier on testing set=',rfc.score(xtest,np.ravel(ytest)))
    xgbc=XGBClassifier()
    xgbc.fit(xtrain,ytrain)
    print('acc using xgboost on testing set=',rfc.score(xtest,np.ravel(ytest)))


