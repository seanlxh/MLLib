import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_classification
class Adaline(object):
    def __init__(self,rate = 0.1,round = 100):
        self.rate = rate
        self.round = round

    def fit(self,X,y):
        self.w_ = np.zeros(1+X.shape[1])
        self.cost_ = []

        for i in range(self.round):
            output = self.predit(X)
            errors = y - output
            self.w_[1:] = self.rate * X.T.dot(errors)
            self.w_[0] = self.rate * errors.sum()
            cost = (errors ** 2).sum()/2.0
            self.cost_.append(cost)
        return self

    def predit(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.predit(X) > 0.0, 1, -1)

# X1 = np.array([[1,3],[2,1],[3,1]])
# Y1 = np.array([1,-1,-1])
X1, Y1 = make_classification(n_samples=200, n_features=2, n_redundant=0,
                             n_clusters_per_class=1, n_classes=2)
TestX = np.array([1,2,3])
TestY = TestX[1:] + TestX[0]
print(TestY)
tmp = Adaline()
obj = tmp.fit(X1,Y1)
print obj.w_
x = np.arange(-1, 4, 0.1)
y = -(obj.w_[1]/obj.w_[2])*x - obj.w_[0]/obj.w_[2]
plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1)
plt.plot(x,y)

plt.show()