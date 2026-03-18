import numpy as np
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression, MeanModel


data = np.loadtxt("Solubilidade.csv",delimiter=',')
X = data[:,:-1]
y = data[:,-1:]
N,p = X.shape

#Validação
R = 500
mse_OLS_COM = []
for r in range(R):
    #Embaralhando  conjunto de dados
    idx = np.random.permutation(N)
    X_aleatorio = np.copy(X)[idx, :]
    y_aleatorio = np.copy(y)[idx, :]
    
    #Particionar o CJ de dados (80/20)
    X_treino = X_aleatorio[:int(N*.8), :]
    y_treino = y_aleatorio[:int(N*.8), :]
    
    X_teste = X_aleatorio[int(N*.8):,:]
    y_teste = y_aleatorio[int(N*.8):,:]
    
    
    #OLS COM INTERCEPTO
    ols = LinearRegression(X_treino, y_treino)
    ols.fit()
    y_hat = ols.predict(X_teste)
    SSE = np.sum((y_teste - y_hat)**2)
    MSE = np.mean((y_teste - y_hat)**2)
    mse_OLS_COM.append(MSE)
    y_bar = np.mean(y_teste)
    SST = np.sum((y_teste - y_bar)**2)
    R2 = 1 - SSE/SST
    print(f"OLS COM INTERCEPTO ")
    print(f"SSE = {SSE:.4f}, MSE = {MSE:.4f}, R^2 = {R2:.4f}")
    
    #OLS SEM INTERCEPTO
    ols = LinearRegression(X_treino, y_treino, fit_intercept=False)
    ols.fit()
    y_hat = ols.predict(X_teste)
    SSE = np.sum((y_teste - y_hat)**2)
    MSE = np.mean((y_teste - y_hat)**2)
    y_bar = np.mean(y_teste)
    SST = np.sum((y_teste - y_bar)**2)
    R2 = 1 - SSE/SST
    print(f"OLS SEM INTERCEPTO ")
    print(f"SSE = {SSE:.4f}, MSE = {MSE:.4f}, R^2 = {R2:.4f}")
    
    
    
    #Modelo da Média
    media = MeanModel(y_treino)
    media.fit()
    y_hat = media.predict(X_teste)
    SSE = np.sum((y_teste - y_hat)**2)
    MSE = np.mean((y_teste - y_hat)**2)
    y_bar = np.mean(y_teste)
    SST = np.sum((y_teste - y_bar)**2)
    R2 = 1 - SSE/SST
    print(f"Média")
    print(f"SSE = {SSE:.4f}, MSE = {MSE:.4f}, R^2 = {R2:.4f}")
    
    # y_hat = media.predict(X_treino)
    # y_bar = np.mean(y_treino)
    # SSE = np.sum((y_treino - y_hat)**2)
    # SST = np.sum((y_treino - y_bar)**2)
    # R2 = 1 - SSE/SST
    # print(R2)
    
    bp=1


print(f"Média de MSE : {np.mean(mse_OLS_COM)}±{np.var(mse_OLS_COM)}")


bp = 1

















'''

p variáveis independentes (preditores, atributos)
C classes (C categorias)
C > 1
'''






