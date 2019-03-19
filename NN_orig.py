import numpy as np

class NN:
    
    def __init__(self,In=1,Hid=1,Out=1,Rate=1):
        import numpy as np
        from scipy.special import expit
        self.In = In
        self.Out = Out
        self.Hid = Hid
        self.Rate = Rate
        self.wih = np.random.normal(0.0, pow(self.Hid, -0.5),(self.Hid, self.In))
        self.who = np.random.normal(0.0, pow(self.Out, -0.5),(self.Out, self.Hid))
        self.activ = lambda x: expit(x)
        
    def query(self,inputs):
        import numpy as np
        from scipy.special import expit
        inputs = np.array(inputs,ndmin = 2).T
        hid_in = np.dot(self.wih,inputs)
        hid_out = self.activ(hid_in)
        fin_in = np.dot(self.who, hid_out)
        fin_out = self.activ(fin_in)
        return fin_out
    
    def _train(self,inputs,targets):
        import numpy as np
        from scipy.special import expit
        inputs = np.array(inputs, ndmin=2).T
        targets = np.array(targets, ndmin=2).T
        
        hid_in = np.dot(self.wih, inputs)
        hid_out = self.activ(hid_in)
        fin_in = np.dot(self.who, hid_out)
        fin_out = self.activ(fin_in)
        
        out_err = targets - fin_out
        hid_err = np.dot(self.who.T, out_err)
        in_err = np.dot(self.wih.T, hid_err)
        
        self.who += self.Rate * np.dot((out_err * fin_out * (1.0 - fin_out)), np.transpose(hid_out))
        self.wih += self.Rate * np.dot((hid_err * hid_out * (1.0 - hid_out)), np.transpose(inputs))
        
    def train(self,datas,targets,epochs = 1,amount=-1):
        import time
        start = time.time()
        if len(datas)!=len(targets):
            raise Exception('Invalid amount.')
        if amount == -1:
            amount = len(datas)
        for x in range(epochs):
            for index,data in enumerate(datas[:amount+1]):
                self._train(data,targets[index])
            print('{} passed to train on {} examples, epoch {}.'.format(time.time()-start,amount,x+1))
        
    def error(self,datas,targets,amount=-1):
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
    
    def save(self,name):
        import zipfile as z
        from os.path import isfile
        from os import remove
        
        if isfile('{}.zip'.format(name)):
            raise Exception('Name exists. Aborted.')
            return
        
        wihname = '{}_wih'.format(name)
        whoname = '{}_who'.format(name)
        np.save(wihname,self.wih)
        np.save(whoname,self.who)
        
        dims = [self.In, self.Out, self.Hid, self.Rate]
        file = open('{}_dims.txt'.format(name),'w')
        for x in dims:
            file.writelines(str(x)+'\n')
        file.close()
        
        zipfile = z.ZipFile('{}.zip'.format(name),'w')
        zipfile.write('{}.npy'.format(wihname))
        zipfile.write('{}.npy'.format(whoname))
        zipfile.write('{}_dims.txt'.format(name))
        zipfile.close()
        
        remove('{}.npy'.format(wihname))
        remove('{}.npy'.format(whoname))
        remove('{}_dims.txt'.format(name))
        
    def load(self,name):
        import zipfile as z
        from os.path import isfile
        from os import remove
        
        if not isfile('{}.zip'.format(name)):
            raise Exception('No such file.')
            return
        
        zipfile = z.ZipFile('{}.zip'.format(name),'r')
        zipfile.extractall()
        
        wihname = '{}_wih.npy'.format(name)
        whoname = '{}_who.npy'.format(name)
        dimname = '{}_dims.txt'.format(name)
        
        self.wih = np.load(wihname)
        self.who = np.load(whoname)
        file = open(dimname,'r')
        
        dims = file.readlines()
        dims = [float(x[:-1]) for x in dims]
        
        self.In = dims[0]
        self.Out = dims[1]
        self.Hid = dims[2]
        self.Rate = dims[3]
        
        remove(wihname)
        remove(whoname)
        remove(dimname)
        
    def show(self, datas, labels, index=0):
        import numpy as np
        
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
        
        image_array = np.asfarray(datas[index]).reshape((28,28))
        plt.imshow(image_array, cmap='Greys', interpolation='None')
