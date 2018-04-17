import numpy as np
from scipy.sparse import csc_matrix

class Model(object):

    def train(self, X, y):
        """ train the model.
        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].
            y: A dense array of ints with shape [num_examples].
        """
        raise NotImplementedError()

    def predict(self, s):
        """ use the model to do prediction.
        Args:
            s: the sample that to be predicted using the model.
        """
        raise NotImplementedError()

class AdaBoost(Model):
    def __init__(self, iterate_number):
        self.iterate_number = iterate_number
        self.H = []

    def train(self, X, y):
        X = X.toarray()
        y = np.mat(y).T
        y[y == 0] = -1

        n, m = X.shape
        D = np.ones((n,1))/n

        for time in range(self.iterate_number):
            h_t, minErr, resarray = self.newHt(X,y,D)
            if minErr > 0.5:
                break
            alpha = float(0.5*np.log((1-minErr)/max(minErr,1e-16)))
            h_t['alpha'] = alpha
            self.H.append(h_t)
            expon = np.multiply(-1*alpha*y,resarray)
            D = np.multiply(D, np.exp(expon)) 
            D = D/D.sum()

    def newHt(self,X, lables, D):
        n, m = X.shape
        ans = {}       
        minErr = np.inf
        resarray = []

        for i in range(m):
            ordered = np.sort(X[:,i])
            
            for j in range(n-1):
                if ordered[j] == ordered[j+1]:
                    continue
                threshold = (ordered[j] + ordered[j])/2
                for tag in ['left-1','left1']:
                    vec = X[:,i]
                    res = self.stump(vec, threshold, tag)

                    errArray = np.mat(np.ones((n,1)))
                    errArray[res == lables] = 0
                    werr = D.T * errArray

                    if werr < minErr:
                        resarray = res
                        minErr = werr
                        ans['hj'] = i
                        ans['hc'] = threshold
                        ans['tag'] = tag                      
        return ans, minErr, resarray

    def stump(self, vec, threshold, tag):
        resultArray = np.ones((vec.shape[0],1))
        if tag == 'left-1':
            resultArray[vec < threshold] = -1
        else:
            resultArray[vec > threshold] = -1

        return resultArray
                    
    
    def predict(self, S):
        dataMatrix = S.toarray()
        n = dataMatrix.shape[0]
        posible = np.zeros((n,1))

        for h in self.H:
            classres = self.stump(dataMatrix[:,h['hj']], h['hc'], h['tag'])
            posible += classres * h['alpha']

        outc = np.zeros((n,1),dtype = int)
        outc[posible >= 0] = 1
        outc[posible < 0] = 0
        outc = np.squeeze(outc)
        return outc




        
