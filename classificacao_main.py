import numpy as np
import matplotlib.pyplot as plt

# 1. Carregamento dos dados
# O arquivo possui 3 linhas e 50000 colunas. 
data_emg = np.loadtxt("EMGsDataset.csv", delimiter=',')

# Linhas 0 e 1 são as características (Sensores). Linha 2 são as classes (1 a 5).
X_raw = data_emg[0:2, :]  # Shape: (2, 50000)
y_raw = data_emg[2, :]    # Shape: (50000,)

N = X_raw.shape[1]
C = 5 # 5 classes de expressões faciais

# 2. Preparação das matrizes para o MQO de Classificação
# O PDF exige X em R^(N x p) e Y em R^(N x C) para o MQO
X_mqo = X_raw.T # Transpõe para ficar (50000, 2)

# Criando a matriz Y (One-Hot Encoding) na mão
Y_mqo = np.zeros((N, C))
for i in range(N):
    classe_atual = int(y_raw[i]) - 1 # Subtrai 1 porque os índices em Python começam em 0
    Y_mqo[i, classe_atual] = 1

print(f"Shape de X para MQO: {X_mqo.shape}")
print(f"Shape de Y para MQO: {Y_mqo.shape}")

# 3. Preparação das matrizes para os Modelos Gaussianos
# O PDF exige X em R^(p x N) e Y em R^(C x N) 
X_gauss = X_raw # Já está no formato (2, 50000)
Y_gauss = Y_mqo.T # Transpõe o Y que criamos para ficar (5, 50000)

print(f"Shape de X para Gaussianos: {X_gauss.shape}")
print(f"Shape de Y para Gaussianos: {Y_gauss.shape}")