from sklearn.preprocessing import StandardScaler
import numpy as np
import inspect

class Predictor:
    def __init__(self,estor):
        self.scaler_X = StandardScaler(with_mean=True,with_std=True)
        self.scaler_y = StandardScaler(with_mean=True,with_std=True)
        self.estor = estor

    def train(self,X,y):
        assert len(X.shape)==3,'in_data must have three dimensions'
        X = np.reshape(X,newshape=(X.shape[0],-1),order='C')
        y = np.reshape(y,newshape=(y.shape[0],1),order='C')

        self.scaler_X.fit(X)
        X = self.scaler_X.transform(X) 

        self.scaler_y.fit(y)
        y = self.scaler_y.transform(y) 

        self.estor.fit(X,y.ravel())
        print('Best params is {}'.format(self.estor.best_params_))

    def predict(self,X):
        assert len(X.shape)==3,'in_data must have three dimensions'
        X = np.reshape(X,newshape=(X.shape[0],-1),order='C')
        X = self.scaler_X.transform(X)
        y = self.estor.predict(X)
        y = np.reshape(y,newshape=(y.shape[0],1),order='C')
        y = self.scaler_y.inverse_transform(y)
        return y.ravel()

    def params(self):
        return self.estor.best_params_

    def score(self,X,y,metric,**kwargs):
        y_pred = self.predict(X)
        sc = metric(y_true=y,y_pred=y_pred,**kwargs)
        return sc
    
    def region_score(self,X,y,metric,reg_scorer,n_repeats=1,summary=True,**kwargs):
        if summary:
            scores = np.zeros(X.shape[1],dtype=np.float32,order='C')
            serial = np.arange(0,X.shape[0],1,dtype=int)
            true_score = self.score(X,y,metric,**kwargs)
            for i in range(n_repeats):
                rndidx = np.random.permutation(serial)  
                for j in range(X.shape[1]):
                    X_rnd = np.copy(X,order='C')
                    X_rnd[:,j] = X[rndidx,j] 
                    rnd_score = self.score(X_rnd,y,metric,**kwargs)
                    scores[j] += rnd_score
            for j in range(X.shape[1]):
                scores[j] = reg_scorer(sc_true=true_score,sc_rnd=scores[j]/n_repeats)
        else:
            scores = np.zeros((X.shape[0],X.shape[1]),dtype=np.float32,order='C')
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    X_rnd = np.copy(X,order='C')
                    X_rnd[:,:j] = X[i:i+1,:j]
                    X_rnd[:,j+1:] = X[i:i+1,j+1:]
                    rnd_score = self.score(X_rnd,y[i:i+1],metric,**kwargs)
                    true_score = self.score(X[i:i+1],y[i:i+1],metric,**kwargs)
                    scores[i,j] = reg_scorer(sc_true=true_score,sc_rnd=rnd_score)
        return scores

