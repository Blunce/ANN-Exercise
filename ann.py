# coding=gbk
'''
Created on 2015年4月21日

@author: Blunce

'''
import numpy as np
import math
import matplotlib.pyplot as plt

ITER_NUM = 100
LEARN_RATE = 0.3
NUM_DATA = 10

class MyANN:
    def __init__(self, input, hidden, output):
        self.inputLayer = input
        self.hiddenLayer = hidden
        self.outputLayer = output
        self.weight1 = np.mat(np.random.random((input, hidden)))
        self.weight2 = np.mat(np.random.random((hidden, output)))
        self.bias1 = np.mat(np.random.random((1, hidden)))
        self.bias2 = np.mat(np.random.random((1, output)))
        self.inputData = np.mat(np.zeros((1, input)))
        self.hiddenData = np.mat(np.zeros((1, hidden)))
        self.outputData = np.mat(np.zeros((1, output)))
    
    def save(self):
        import pickle
        with open('network.txt', 'w') as fw:
            pickle.dump(self, fw)
            
    def load(self, filename):
        import pickle
        with open(filename) as fr:
            pickle.load(self, fr)
            
    def actionFunction(self, x):
        return 1.0 / (1.0 + math.e ** (-x))
    
    def test(self):
        self.weight1 = np.mat([[0.2, -0.3], [0.4, 0.1], [-0.5, 0.2]])
        self.weight2 = np.mat([[-0.3], [-0.2]])
        self.bias1 = np.mat([[-0.4, 0.2]])
        self.bias2 = np.mat([[0.1]])
        
    def calculateOne(self, onetuple):
        self.inputData = np.mat(onetuple)
        self.hiddenData = np.mat(self.actionFunction(np.array(self.inputData * self.weight1 + self.bias1)))
        self.outputData = np.mat(self.actionFunction(np.array(self.hiddenData * self.weight2 + self.bias2)))
    
def BackPropagation(net, data, label, LEARN_RATE, ITER_NUM):
    tag = []
    for item in range(ITER_NUM):
        realError = 0
        for i in range(len(data)):
            tuple , aim = np.mat(data[i]), np.mat(label[i])
            net.calculateOne(tuple)
            realError += np.mean(np.array(aim - net.outputData))
            outputErr = np.array(aim - net.outputData) * (1 - net.outputData) * np.array(net.outputData)
            hiddenErr = np.array(net.hiddenData) * np.array((1 - net.hiddenData)) * np.array(outputErr * net.weight2.T)
            
            net.weight1 += LEARN_RATE * tuple.T * np.mat(hiddenErr)
            net.weight2 += LEARN_RATE * net.hiddenData.T * np.mat(outputErr)
            
            net.bias1 += LEARN_RATE * np.mat(hiddenErr)
            net.bias2 += LEARN_RATE * np.mat(outputErr)
        tag.append(realError/len(data))
        '''    
        error = 0
        for i in range(len(data)):
            net.calculateOne(np.mat(data[i]))
            error += (((np.array(net.outputData) - label[i]) ** 2).sum() / len(label[i])) ** 0.5
        tag.append(error / len(data))'''
    return np.array(tag)

def f(x, y):
    return math.e ** x / (y ** 2 + 1)

def loadData():
    import random
    data, label = [], []
    for i in range(NUM_DATA):
        x, y = random.uniform(-10, 10), random.uniform(-10, 10)
        data.append((x, y))
        label.append([f(x, y)])
    return np.mat(data), np.mat(label)

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(maxVals, (m, 1))
    return normDataSet, ranges, minVals

if __name__ == '__main__':
    # for test
    net = MyANN(2, 4, 1)
    data, label = loadData()
    normData, rangesData, minValsData = autoNorm(data)
    normLabel, rangesLabel, minValsLabel = autoNorm(label)

    y = BackPropagation(net, normData, normLabel, LEARN_RATE, ITER_NUM)
    print 'finish traning'
    x = np.arange(ITER_NUM)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y)
    fig.savefig('test.pdf')
    print 'finish drawing'
    
