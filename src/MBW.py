# An implementation of the Modified Balanced Winnow algorithm

import data
import src.config
import numpy as np
import pickle



class MBWinnow(object):
    def __init__(self,size = 20000,theta = 1 ,M = 1,alpha = 1.5,beta = 0.5):
        self.theta = theta
        self.M = M
        self.alpha = alpha
        self.beta = beta
        self.w_plus = np.zeros(size) + 2
        self.w_minus = np.zeros(size) + 1
        self.sum_w_plus = np.zeros(size) # we keep the sum of all w's for voting
        self.sum_w_minus = np.zeros(size)
        self.corrects_since_last = 0
        self.z = 0 # sum of correct classifications

    def check(self,xt,yt): # check if xt is misclassified with respect to yt
        if yt != 0 : # 0 normal. anything else attack
            yt = -1
        else:
            yt = 1
        # the paper requires us to normalize each input vector
        norm = np.linalg.norm(xt)
        if norm != 0:
            xt = xt / norm
        score = np.dot(xt,self.w_plus) - np.dot(xt,self.w_minus) - self.theta

        if score * yt < self.M:
            return False,xt,yt
        return True,None,None

    def update(self,xt,yt):
        # print("update")
        if yt > 0:
            for i,x in enumerate(xt):
                if x > 0 :
                    self.w_plus[i] = self.w_plus[i] * self.alpha * (1 + x)
                    self.w_minus[i] = self.w_minus[i] * self.beta * (1 - x)
        else:
            for i,x in enumerate(xt):
                if x > 0 :
                    self.w_plus[i] = self.w_plus[i] * self.beta * (1 - x)
                    self.w_minus[i] = self.w_minus[i] * self.alpha * (1 + x)
        #self.validate()
        self.z += self.corrects_since_last
        self.sum_w_plus += self.corrects_since_last * self.w_plus
        self.sum_w_minus += self.corrects_since_last * self.w_minus
        self.corrects_since_last = 0


    def train(self,X_train,y_train): # either train with data from controller or with X_train
        for i, x in enumerate(X_train):
            result, x, y = self.check(x, y_train[i])
            if result is False:
                self.update(x, y)

    def predict(self,x):
        norm = np.linalg.norm(x)
        if norm != 0:
            x = x / norm
        plus_vector = self.sum_w_plus/self.z
        minus_vector = self.sum_w_minus/self.z
        score = np.dot(x, plus_vector) - np.dot(x, minus_vector) - self.theta
        if score < 0:
            return -1
        else:
            return 1

    def evaluate(self,X_test,y_test,toFile = True, path = 'results.txt',verbose = True): # evaluates the model on test data
        #  and prints the desired parameters
        tp = 0
        fp = 0
        fn = 0
        total = 0
        for i, x in enumerate(X_test):
            y = y_test[i]
            if y == 0:
                y = 1
            else:
                y = -1
            predicted = self.predict(x)
            if predicted == y:
                if y == 1:
                    tp += 1
            else:
                if y == 1:
                    fn += 1
                else:
                    fp += 1
        precision = tp/(tp + fp)
        recall = tp/(tp + fn)
        f1 = (2 * precision * recall)/(recall + precision)
        if verbose:
            print("*************************************")
            print("Model with alpha = ",self.alpha," beta = ",self.beta, " M = ",self.M)
            print("Precision : ",precision)
            print("Recall : ",recall)
            print("F1 score : ",f1)
        if toFile:
            with open (path,'w') as results:
                results.write("*************************************\n")
                print("Model with alpha = ", self.alpha, " beta = ", self.beta , " M = ",self.M)
                results.write("Precision : " + str(precision) + '\n')
                results.write("Recall : " + str(recall) + '\n')
                results.write("F1 Score : " + str(f1) + '\n')
        return f1,precision, recall
    def save(self, path = 'MBW.pkl'): #saves the current object in a file
        with open(path , 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)







