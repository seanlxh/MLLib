# -*- coding: UTF-8 -*-
import numpy as np
class WeakClassifier:
    def __init__(self,X,Y,W):
        self.X = X
        self.Y = Y
        self.W = W

    def stumpprocess(self,X,i,threshold,ineq):
        pred = np.ones(X.shape[1])
        if ineq == 1:
            pred[X[i:] > threshold] = -1
        else:
            pred[X[i:] <= threshold] = -1
        return pred



    def buildstump(self,step_num):
        row,column= self.X.shape
        dump_label = np.zeros(column)
        dump_tree = {}
        for i in row:
            min_num = self.X[i,:].min()
            max_num = self.X[i,:].max()
            step_size = (max_num - min_num) / step_num
            minerror = np.Inf
            for j in range(0 , step_num):
                threshold = min_num + j*step_size
                for k in range(1,2):
                    pred = self.stumpprocess(self,self.X,i,threshold,k)
                    errLabel = np.zeros(self.X.shape[1])
                    errLabel[pred == self.Y] = 1
                    weightederr = self.W.dot(errLabel)
                    if weightederr < minerror:
                        minerror = weightederr
                        bestclass = pred









