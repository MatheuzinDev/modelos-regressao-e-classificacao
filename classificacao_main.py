import numpy as np
import matplotlib.pyplot as plt

data_emg = np.loadtxt("EMGsDataset.csv", delimiter=',')

X_raw = data_emg[0:2, :]  
y_raw = data_emg[2, :]    

N = X_raw.shape[1]
C = 5 

# 1. Preparando as matrizes para o MQO de Classificação
X_mqo = X_raw.T 

# Aqui nós realizamos o One-Hot Encoding.
Y_mqo = np.zeros((N, C))
for i in range(N):
    classe_atual = int(y_raw[i]) - 1 
    Y_mqo[i, classe_atual] = 1

print(f"Shape de X para MQO: {X_mqo.shape}")
print(f"Shape de Y para MQO: {Y_mqo.shape}")

# 2. Preparando as matrizes para os Modelos Gaussianos
X_gauss = X_raw 
Y_gauss = Y_mqo.T 

print(f"Shape de X para Gaussianos: {X_gauss.shape}")
print(f"Shape de Y para Gaussianos: {Y_gauss.shape}")