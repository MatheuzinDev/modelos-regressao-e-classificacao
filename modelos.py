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
        """
        X_train: (N, p)
        Y_train: (N, C) - matriz one-hot
        """
        self.X_train = X_train
        self.Y_train = Y_train
        self.fit_intercept = fit_intercept
        self.N, self.p = X_train.shape
        
        # Adiciona a coluna de 1s para o intercepto
        if fit_intercept:
            self.X_train = np.hstack((np.ones((self.N, 1)), X_train))
        
        self.W_hat = None # Matriz de pesos

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
            
        # Calcula as estimativas para todas as classes
        Y_pred_continuo = X_test_mod @ self.W_hat
        
        # A classe predita é a coluna com o maior valor (retorna o índice 0 a 4)
        classes_preditas = np.argmax(Y_pred_continuo, axis=1)
        
        # Retornamos +1 para voltar ao padrão original das classes (1 a 5)
        return classes_preditas + 1

class ClassificadorGaussianoTradicional:
    def __init__(self):
        self.classes = None
        self.priori = {}
        self.medias = {}
        self.covariancias = {}

    def fit(self, X_train, Y_train):
        """
        X_train: (p, N) - Ex: (2, 50000)
        Y_train: (C, N) - Ex: (5, 50000)
        """
        p, N = X_train.shape
        C = Y_train.shape[0]
        self.classes = range(C)
        
        for c in self.classes:
            # Encontra os índices onde a classe atual é 1
            indices_c = np.where(Y_train[c, :] == 1)[0]
            Nc = len(indices_c)
            
            # Pega apenas as amostras (colunas) da classe c
            X_c = X_train[:, indices_c]
            
            # 1. Probabilidade a priori: Nc / N
            self.priori[c] = Nc / N
            
            # 2. Vetor de Médias (shape: p x 1)
            # Calcula a média nas linhas (eixo 1) e mantém as dimensões
            mu_c = np.mean(X_c, axis=1, keepdims=True)
            self.medias[c] = mu_c
            
            # 3. Matriz de Covariância (shape: p x p)
            # Equação: (1/Nc) * (X_c - mu_c) @ (X_c - mu_c).T
            X_c_centralizado = X_c - mu_c
            cov_c = (1 / Nc) * (X_c_centralizado @ X_c_centralizado.T)
            self.covariancias[c] = cov_c

    def predict(self, X_test):
        """
        X_test: (p, N_test)
        Retorna um vetor (N_test,) com as classes preditas de 1 a 5
        """
        p, N_test = X_test.shape
        C = len(self.classes)
        
        # Matriz para guardar o "score" de cada classe para cada amostra
        # Shape (C, N_test)
        discriminantes = np.zeros((C, N_test))
        
        for c in self.classes:
            mu_c = self.medias[c]
            cov_c = self.covariancias[c]
            priori_c = self.priori[c]
            
            # Estabilização numérica (Jitter) para evitar Matriz Singular
            epsilon = 1e-6
            cov_c_estavel = cov_c + np.eye(cov_c.shape[0]) * epsilon
            
            # Inversa e Determinante da Covariância usando a matriz estabilizada
            inv_cov = np.linalg.inv(cov_c_estavel)
            det_cov = np.linalg.det(cov_c_estavel)
            
            # Centralizando as amostras de teste em relação à média da classe c
            X_centralizado = X_test - mu_c  # shape (p, N_test)
            
            # Cálculo do termo quadrático para todas as amostras de uma vez:
            # diag( X_cent^T @ inv_cov @ X_cent )
            termo_quadratico = np.sum(X_centralizado * (inv_cov @ X_centralizado), axis=0)
            
            # Função Discriminante Gaussiana
            # g_c(x) = -0.5 * termo_quadratico - 0.5 * ln(|cov|) + ln(priori)
            g_c = -0.5 * termo_quadratico - 0.5 * np.log(det_cov) + np.log(priori_c)
            
            discriminantes[c, :] = g_c
            
        # A classe escolhida é a que tem o maior valor na função discriminante
        # argmax no eixo 0 (olhando pelas linhas)
        classes_preditas = np.argmax(discriminantes, axis=0)
        
        # Soma 1 para voltar ao padrão de rótulos originais (1, 2, 3, 4, 5)
        return classes_preditas + 1

class ClassificadorGaussianoCovGlobal(ClassificadorGaussianoTradicional):
    def fit(self, X_train, Y_train):
        # 1. Roda o treinamento tradicional para achar médias e prioris
        super().fit(X_train, Y_train)
        
        # 2. Calcula a covariância global ignorando as classes (X_train inteiro)
        mu_global = np.mean(X_train, axis=1, keepdims=True)
        X_cent_global = X_train - mu_global
        N_total = X_train.shape[1]
        cov_global = (1 / N_total) * (X_cent_global @ X_cent_global.T)
        
        # 3. Força todas as classes a usarem essa mesma matriz global
        for c in self.classes:
            self.covariancias[c] = cov_global

class ClassificadorGaussianoCovAgregada(ClassificadorGaussianoTradicional):
    def fit(self, X_train, Y_train):
        super().fit(X_train, Y_train)
        
        # A covariância agregada é a média ponderada das covariâncias de cada classe
        cov_agregada = np.zeros_like(self.covariancias[0])
        for c in self.classes:
            cov_agregada += self.priori[c] * self.covariancias[c]
            
        # Força todas as classes a usarem a matriz agregada (Pooled)
        for c in self.classes:
            self.covariancias[c] = cov_agregada

class ClassificadorNaiveBayes(ClassificadorGaussianoTradicional):
    def fit(self, X_train, Y_train):
        super().fit(X_train, Y_train)
        
        # No Naive Bayes, assumimos que as variáveis são independentes.
        # Matematicamente, isso significa zerar tudo que está fora da diagonal principal da covariância.
        for c in self.classes:
            cov_c = self.covariancias[c]
            # np.diag extrai a diagonal, o segundo np.diag recria a matriz só com a diagonal
            cov_nb = np.diag(np.diag(cov_c)) 
            self.covariancias[c] = cov_nb

class ClassificadorGaussianoRegularizado(ClassificadorGaussianoTradicional):
    def __init__(self, lamb=0.0):
        super().__init__()
        self.lamb = lamb  # Hiperparâmetro de Friedman

    def fit(self, X_train, Y_train):
        super().fit(X_train, Y_train)
        
        # 1. Primeiro calcula a covariância agregada
        cov_agregada = np.zeros_like(self.covariancias[0])
        for c in self.classes:
            cov_agregada += self.priori[c] * self.covariancias[c]
            
        # 2. A regularização de Friedman é a interpolação entre a covariância da classe e a agregada
        for c in self.classes:
            # Se lamb=0, é o Tradicional. Se lamb=1, é o CovAgregada.
            self.covariancias[c] = (1 - self.lamb) * self.covariancias[c] + self.lamb * cov_agregada

bp = 1