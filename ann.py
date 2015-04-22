# coding=gbk
'''
Created on 2015年4月21日

@author: Blunce

'''
import numpy as np
import math

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
    
    def calculateOne(self, onetuple):
        self.inputData = np.mat(onetuple)
        self.hiddenData = np.mat(self.actionFunction(np.array(self.inputData * self.weight1 + self.bias1)))
        self.outputData = np.mat(self.actionFunction(np.array(self.hiddenData * self.weight2 + self.bias2)))
    
def BackPropagation(net, data, label, learn_rate, iterNum):
    tag = []
    for item in range(iterNum):
        for i in range(len(data)):
            tuple , aim = np.mat(data[i]), np.mat(label[i])
            net.calculate(tuple)
            outputErr = np.array(aim - net.outputData)*(1-aim)*np.array(aim)
            hiddenErr = np.array(net.hiddenData)*(1-net.hiddenData)*np.array(outputErr*net.weight2.T)
        error = 0
        for i in range(len(data)):
            net.calculate(np.mat(data[i]))
            error += (((np.array(net.outputData)-label[i])**2).sum()/len(label[i]))**0.5
        tag.append(error/len(data))
    return tag

def loadData():
    pass
    
