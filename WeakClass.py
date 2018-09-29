# -*- coding: UTF-8 -*-
import numpy as np
class WeakClassifier:
    def __init__(self,X,Y,W):
        self.X = X
        self.Y = Y
        self.W = W

    def buildstump(self,step_num):
        N ,d= self.X.shape
        dump_label = np.zeros(N)
        dump_tree = {}
        print N,d






