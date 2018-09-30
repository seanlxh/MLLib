# -*- coding:utf-8 -*-
from WeakClass import *
from numpy import np

#Adaboost类
class Adaboost:
    weight = {}
    classfies = {}
    ##训练一个分类器，并将分类器及其权重保存到字典中
    def train(self,X,Y,train_name,round=100,step_num=20):
        dataset_size = X.shape[0]
        one_weight = np.zeros(dataset_size,dtype='float')
        one_weight = 1.0/dataset_size;
        classfier = WeakClassifier(X,Y,one_weight)
        for i in range(0,round):
            dump_label, dump_tree, min_error = classfier.buildstump(step_num)
            alpha = 1.0/2*np.log2((1.0 - min_error)/min_error)
            Z = 0
            for j in range(0,i):
                Z = Z + one_weight[j]*np.exp(-alpha*Y[j]*self.classfies[train_name][j].prediction(X)[0])
            # np.dot(np.exp(-alpha*Y*)

        return
    ##预测
    def prediction(self,X,Y):




        return


