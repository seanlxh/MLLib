import numpy as np
class mlp(object):
    def sigmod(z):
        return 1.0 / (1.0 + np.exp(-z))

    def __init__(self, rate=0.1, epoch = 100 , size = None):
        self.rate = rate;
        self.maxEpoch = epoch
        self.size = size
        self.W = []
        self.b = []
        self.init()

    def forwardPropagation(self,item = None):
        a = [item]
        for wIndex in range(len(self.W)):
            a.append(self.sigmod(self.W[wIndex] * a[-1] + self.b[wIndex]))

        return a

    def backPropagation(self, label=None, a=None):
        tag = [(a[-1] - label) * a[-1] * (1.0 - a[-1])]
        for i in range(len(self.W) - 1):
            one = np.multiply(a[-2-i], 1-a[-2-i])
            two = np.multiply(self.W[-1-i].T*tag[-1] , one)
            tag.append(two)
        for j in range(len(tag)):
            ads = tag[j] * a[-2-j].T
            self.W[-1-j] = self.W[-1-j]-self.rate*(ads)
            self.b[-1-j] = self.b[-1-j]-self.rate*tag[j]

        error = 0.5*(a[-1] - label)**2
        return error
    def init(self):
        for i in range(len(self.size) - 1):
            self.W.append(np.mat(np.random.uniform(-1,1,size=(self.size[i+1],self.size[i]))))
            self.b.append(np.mat(np.random.uniform(-1,-1,size=(self.size[i+1],1))))

    def train(self, input_=None, target=None, show=10):
        for i in range(self.maxEpoch):
            error = []
            for itemIndex in range(input_.shape[1]):
                a = self.forwardPropagation(input_[:,itemIndex])
                e = self.backPropagation(target[:,itemIndex],a)
                error.append(e[0,0])
            ratio = sum(error) / len(error)
            if i % show == 0:
                print("轮次 {0}: ".format(i), ratio)

    def predict(self, ins=None):
        return self.forwardPropagation(item=ins)[-1]
