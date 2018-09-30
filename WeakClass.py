# -*- coding: UTF-8 -*-
import numpy as np
class WeakClassifier:
    def __init__(self,X,Y,W):
        self.X = X
        self.Y = Y
        self.W = W

    def stumpprocess(self,X,i,threshold,ineq):
        pred = np.ones(X.shape[0])
        if ineq == 1:
            pred[X[:,i] > threshold] = -1
        else:
            pred[X[:,i] <= threshold] = -1
        return pred



    def buildstump(self,step_num):
        row,column= self.X.shape
        dump_label = np.zeros(column)
        dump_tree = {}
        min_error = np.Inf
        for i in range(0,column-1):
            min_num = self.X[:,i].min()
            max_num = self.X[:,i].max()
            step_size = (max_num - min_num) / step_num
            for j in range(0 , step_num):
                threshold = min_num + j*step_size
                for k in range(1,3):
                    pred = self.stumpprocess(self.X,i,threshold,k)
                    errLabel = np.zeros(self.X.shape[0])
                    errLabel[pred != self.Y] = 1
                    weightederr = self.W.dot(errLabel)
                    if weightederr < min_error:
                        min_error = weightederr
                        dump_label = pred
                        dump_tree["index"] = i
                        dump_tree["threshold"] = threshold
        return dump_label,dump_tree,min_error


tmp = WeakClassifier(np.array([[1,1],[1,3],[1,5]]),np.array([1,1,1]),np.array([1.0/3,1.0/3,1.0/3],dtype = 'float'))
print tmp.buildstump(3)



