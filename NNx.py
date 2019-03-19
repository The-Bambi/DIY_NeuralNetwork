import numpy as np
from scipy.special import expit
from List import List

class NNx:
    
    def __init__(self,sizes,Rate):
        self.In = sizes[0]
        self.Out = sizes[-1]
        self.Hids = sizes[1:-1]
        self.Rate = Rate
        self.weights = List()
        for w,x in enumerate(sizes[:-1]):
            self.weights.append(np.random.normal(0.0, pow(x, -0.5),(sizes[w+1],sizes[w])))
        self.activ = lambda x: expit(x)
        
    def query(self,inputs):
        inputs = np.array(inputs,ndmin = 2).T
        hid_in = np.dot(self.weights.left.data,inputs)
        hid_out = self.activ(hid_in)
        right = self.weights.left.right
        while right is not None:
            hid_in = np.dot(right.data,hid_out)
            hid_out = self.activ(hid_in)
            right = right.right
        return hid_out
    
    def _train(self,inputs,targets):
        inputs = np.array(inputs, ndmin=2).T
        targets = np.array(targets, ndmin=2).T
        
        outs = List()
        
        hid_in = np.dot(self.weights.left.data,inputs)
        hid_out = self.activ(hid_in)
        outs.append(hid_out)

        right = self.weights.left.right
        while right is not None:
            hid_in = np.dot(right.data,hid_out)
            hid_out = self.activ(hid_in)
            outs.append(hid_out)
            right = right.right
        
        errors = List()
        errors.append(targets-outs.right.data)
        err = errors.right
        wght = self.weights.right
        while wght is not None:
            errors.lappend(np.dot(wght.data.T,err.data))
            wght = wght.left
            err = err.left
            
        outs.lappend(inputs)
        
        wght = self.weights.right
        out = outs.right
        err = errors.right
        while wght is not None:
            wght.data += self.Rate * np.dot((err.data * out.data * (1.0 - out.data)),np.transpose(out.left.data))
            wght = wght.left
            out = out.left
            err = err.left
        
    def train(self,datas,targets,epochs = 1,amount=-1):
        #import time
        #start = time.time()
        if len(datas)!=len(targets):
            raise Exception('Invalid amount.')
        if amount == -1:
            amount = len(datas)
        for x in range(epochs):
            for index,data in enumerate(datas[:amount+1]):
                self._train(data,targets[index])
            #print('{} passed to train on {} examples, epoch {}.'.format(time.time()-start,amount,x+1))
        
    def error(self, datas, targets, amount=-1):
        if len(datas)!=len(targets):
            raise Exception('Invalid amount.')
        if amount == -1:
            amount = len(datas)
        wrong = 0
        right = 0
        for index,data in enumerate(datas[:amount]):
            guess = self.query(data)
            label = targets[index]
            topguess = 0
            toplabel = 0
            for g,h in enumerate(guess):
                if guess[topguess]<h:
                    topguess = g
            for g,h in enumerate(label):
                if label[toplabel]<h:
                    toplabel = g
            if toplabel != topguess:
                wrong += 1
            else: right += 1
        return (right,wrong)
        
    def show(self, datas, labels, dims, index=0):
        numbers = [0,1,2,3,4,5,6,7,8,9]
        guess = self.query(datas[index])
        label = labels[index]
        topguess = 0
        toplabel = 0
        for g,h in enumerate(guess):
            if guess[topguess]<h:
                topguess = g
        for g,h in enumerate(label):
            if label[toplabel]<h:
                toplabel = g
        if toplabel == topguess:
            print('Correct! Guess is {}.'.format(numbers[topguess]))
        else: print('Nope. Guess is {}.'.format(numbers[topguess]))
        
        image_array = np.asfarray(datas[index]).reshape(dims)
        plt.imshow(image_array, cmap='Greys', interpolation='None')
