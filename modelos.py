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

class MQOClassificacao:
    def __init__(self, X_train, Y_train, fit_intercept=True):
        self.X_train = X_train
        self.Y_train = Y_train
        self.fit_intercept = fit_intercept
        self.N, self.p = X_train.shape
        
        # Aqui nós adicionamos as colunas de 1s para intercepto
        if fit_intercept:
            self.X_train = np.hstack((np.ones((self.N, 1)), X_train))
        
        self.W_hat = None 

    # Treinamento
    def fit(self):
        # A mesma equação normal do MQO: W = (X^T * X)^-1 * X^T * Y
        # A diferença é que Y agora é uma matriz, então W será uma matriz
        self.W_hat = np.linalg.inv(self.X_train.T @ self.X_train) @ self.X_train.T @ self.Y_train

    # Predição
    def predict(self, X_test):
        N_test = X_test.shape[0]
        
        if self.fit_intercept:
            X_test_mod = np.hstack((np.ones((N_test, 1)), X_test))
        else:
            X_test_mod = X_test
            
        Y_pred_continuo = X_test_mod @ self.W_hat
        
        classes_preditas = np.argmax(Y_pred_continuo, axis=1)
        
        return classes_preditas + 1

class ClassificadorGaussianoTradicional:
    def __init__(self):
        self.classes = None
        self.priori = {}
        self.medias = {}
        self.covariancias = {}

    def fit(self, X_train, Y_train):
        p, N = X_train.shape
        C = Y_train.shape[0]
        self.classes = range(C)
        
        for c in self.classes:
            
            indices_c = np.where(Y_train[c, :] == 1)[0]
            Nc = len(indices_c)
            
            X_c = X_train[:, indices_c]
            
            # 1. Aplicamos a probabilidade a priori
            self.priori[c] = Nc / N
            
            # 2. Separamos o vetor de médias
            mu_c = np.mean(X_c, axis=1, keepdims=True)
            self.medias[c] = mu_c
            
            # 3. Por último a matriz de covariância
            X_c_centralizado = X_c - mu_c
            cov_c = (1 / Nc) * (X_c_centralizado @ X_c_centralizado.T)
            self.covariancias[c] = cov_c

    def predict(self, X_test):

        p, N_test = X_test.shape
        C = len(self.classes)
        
        discriminantes = np.zeros((C, N_test))
        
        for c in self.classes:
            mu_c = self.medias[c]
            cov_c = self.covariancias[c]
            priori_c = self.priori[c]
            
            # Aqui nós realizamos uma estabilização numérica para evitar casos de matriz singular
            epsilon = 1e-6
            cov_c_estavel = cov_c + np.eye(cov_c.shape[0]) * epsilon
            
            inv_cov = np.linalg.inv(cov_c_estavel)
            det_cov = np.linalg.det(cov_c_estavel)
            
            X_centralizado = X_test - mu_c  
            
            termo_quadratico = np.sum(X_centralizado * (inv_cov @ X_centralizado), axis=0)
            
            g_c = -0.5 * termo_quadratico - 0.5 * np.log(det_cov) + np.log(priori_c)
            
            discriminantes[c, :] = g_c

        classes_preditas = np.argmax(discriminantes, axis=0)
        
        return classes_preditas + 1

class ClassificadorGaussianoCovGlobal(ClassificadorGaussianoTradicional):
    def fit(self, X_train, Y_train):
        super().fit(X_train, Y_train)
        
        mu_global = np.mean(X_train, axis=1, keepdims=True)
        X_cent_global = X_train - mu_global
        N_total = X_train.shape[1]
        cov_global = (1 / N_total) * (X_cent_global @ X_cent_global.T)
        
        for c in self.classes:
            self.covariancias[c] = cov_global

class ClassificadorGaussianoCovAgregada(ClassificadorGaussianoTradicional):
    def fit(self, X_train, Y_train):
        super().fit(X_train, Y_train)
        
        cov_agregada = np.zeros_like(self.covariancias[0])
        for c in self.classes:
            cov_agregada += self.priori[c] * self.covariancias[c]
            
        for c in self.classes:
            self.covariancias[c] = cov_agregada

class ClassificadorNaiveBayes(ClassificadorGaussianoTradicional):
    def fit(self, X_train, Y_train):
        super().fit(X_train, Y_train)
        
        for c in self.classes:
            cov_c = self.covariancias[c]
            cov_nb = np.diag(np.diag(cov_c)) 
            self.covariancias[c] = cov_nb

class ClassificadorGaussianoRegularizado(ClassificadorGaussianoTradicional):
    def __init__(self, lamb=0.0):
        super().__init__()
        self.lamb = lamb  # Hiperparâmetro de Friedman

    def fit(self, X_train, Y_train):
        super().fit(X_train, Y_train)
        
        cov_agregada = np.zeros_like(self.covariancias[0])
        for c in self.classes:
            cov_agregada += self.priori[c] * self.covariancias[c]
            
        for c in self.classes:
            self.covariancias[c] = (1 - self.lamb) * self.covariancias[c] + self.lamb * cov_agregada

bp = 1