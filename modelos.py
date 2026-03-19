import numpy as np

'''
Sempre que eu escrever aqui, eu sou o paulo desenvolvedor da IA
'''
class MeanModel:
    def __init__(self, y_train):
        self.y_train = y_train
        
    # Treinamento
    def fit(self):
        self.beta_0 = np.mean(self.y_train)
        
    # Predição
    def predict(self,X_test):
        N = X_test.shape[0]
        if len(X_test.shape) > 2:
            return np.ones((N,N))*self.beta_0
        else:
            return np.ones((N,1))*self.beta_0
            

class LinearRegression:
    def __init__(self, X_train, y_train, fit_intercept = True):
        
        self.X_train = X_train
        self.y_train = y_train
        self.fit_intercept = fit_intercept
        self.N, self.p = X_train.shape
        if fit_intercept:
            self.X_train = np.hstack((
                np.ones((self.N,1)), X_train
            ))
        
        self.beta_hat = None

    # Treinamento
    def fit(self):
        self.beta_hat = np.linalg.inv(self.X_train.T@self.X_train)@self.X_train.T@self.y_train
        bp=1
        
    # Predição
    def predict(self, X_test):
        N = X_test.shape[0]
        if self.fit_intercept:
            if len(X_test.shape)>2:
                X_test = np.concatenate((
                    np.ones((N,N,1)), X_test
                ),axis = 2)
            else:
                X_test = np.hstack((
                    np.ones((N,1)), X_test
                ))
        
        return X_test @ self.beta_hat
    

        
class RidgeRegression:
    def __init__(self, X_train, y_train, lamb=0.0, fit_intercept=True):
        self.X_train = X_train
        self.y_train = y_train
        self.lamb = lamb
        self.fit_intercept = fit_intercept

        self.N, self.p = X_train.shape

        if self.fit_intercept:
            self.X_train = np.hstack((
                np.ones((self.N, 1)),
                X_train
            ))

        self.beta_hat = None

    # Treinamento
    def fit(self):
        n_features = self.X_train.shape[1]

        I = np.eye(n_features)

        # Não regulariza o intercepto
        if self.fit_intercept:
            I[0, 0] = 0

        self.beta_hat = np.linalg.inv(
            self.X_train.T @ self.X_train + self.lamb * I
        ) @ self.X_train.T @ self.y_train

    # Predição
    def predict(self, X_test):
        N = X_test.shape[0]

        if self.fit_intercept:
            if len(X_test.shape) > 2:
                X_test = np.concatenate((
                    np.ones((N, N, 1)),
                    X_test
                ), axis=2)
            else:
                X_test = np.hstack((
                    np.ones((N, 1)),
                    X_test
                ))

        return X_test @ self.beta_hat







bp = 1