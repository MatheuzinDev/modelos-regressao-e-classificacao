import numpy as np
import matplotlib.pyplot as plt
from modelos import LinearRegression, MeanModel, RidgeRegression

'''
Sempre que eu escrever aqui, eu sou o paulo USUÁRIO da IA
'''

# Carregar os dados do arquivo .dat
data = np.loadtxt("aerogerador.dat")

# Coluna 0 = velocidade do vento (X)
# Coluna 1 = potência gerada (y)
X = data[:, 0:1]
y = data[:, 1:2]

x_plot = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)

# Ajuste do modelo MQO tradicional
lr = LinearRegression(X, y, fit_intercept=False)
lr.fit()
y_plot_lr = lr.predict(x_plot)

# Ajuste do modelo MQO regularizado
ridge = RidgeRegression(X, y, lamb=10000000000, fit_intercept=False)
ridge.fit()
y_plot_ridge = ridge.predict(x_plot)

# Ajuste do modelo da média
modelo_media = MeanModel(y)
modelo_media.fit()
y_plot_media = modelo_media.predict(x_plot)

# Figura 1 - apenas dispersão
# plt.figure(figsize=(8, 5))
# plt.scatter(X[:, 0], y[:, 0], c='magenta', edgecolor='k', alpha=0.7, label='Dados')
# plt.xlabel("Velocidade do vento")
# plt.ylabel("Potência gerada")
# plt.title("Gráfico de espalhamento - Aerogerador")
# plt.legend()
# plt.grid(True)

# Figura 2 - MQO tradicional
plt.figure(figsize=(8, 5))
plt.scatter(X[:, 0], y[:, 0], c='magenta', edgecolor='k', alpha=0.7, label='Dados')
plt.plot(x_plot[:, 0], y_plot_lr[:, 0], linewidth=2, label='MQO tradicional')
plt.xlabel("Velocidade do vento")
plt.ylabel("Potência gerada")
plt.title("MQO tradicional")
plt.legend()
plt.grid(True)

# Figura 3 - MQO regularizado
plt.figure(figsize=(8, 5))
plt.scatter(X[:, 0], y[:, 0], c='magenta', edgecolor='k', alpha=0.7, label='Dados')
plt.plot(x_plot[:, 0], y_plot_ridge[:, 0], linewidth=2, label='MQO regularizado (λ=0.25)')
plt.xlabel("Velocidade do vento")
plt.ylabel("Potência gerada")
plt.title("MQO regularizado")
plt.legend()
plt.grid(True)

# Figura 4 - Modelo da média
# plt.figure(figsize=(8, 5))
# plt.scatter(X[:, 0], y[:, 0], c='magenta', edgecolor='k', alpha=0.7, label='Dados')
# plt.plot(x_plot[:, 0], y_plot_media[:, 0], linewidth=2, label='Modelo da média')
# plt.xlabel("Velocidade do vento")
# plt.ylabel("Potência gerada")
# plt.title("Modelo da média")
# plt.legend()
# plt.grid(True)

# Figura 5 - Histograma dos resíduos
# plt.figure(figsize=(8, 5))
# epsilon = y - lr.predict(X)
# plt.hist(epsilon, edgecolor='k')
# plt.title(f'Resíduos do MQO tradicional | Média: {np.mean(epsilon):.4f}')
# plt.xlabel("Erro")
# plt.ylabel("Frequência")
# plt.grid(True)

# Mostrar tudo ao mesmo tempo
plt.show()