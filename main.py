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

# Visualização inicial dos dados
plt.figure(figsize=(8, 5))
plt.scatter(X[:, 0], y[:, 0], c='magenta', edgecolor='k', alpha=0.7, label='Dados')
plt.xlabel("Velocidade do vento")
plt.ylabel("Potência gerada")
plt.title("Gráfico de espalhamento - Aerogerador")

# Ajuste do modelo OLS com intercepto
lr = LinearRegression(X, y, fit_intercept=True)
lr.fit()

x_plot = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
y_plot_lr = lr.predict(x_plot)

plt.plot(x_plot[:, 0], y_plot_lr[:, 0], label='MQO tradicional', linewidth=2)

# Ajuste do modelo regularizado
ridge = RidgeRegression(X, y, lamb=0.25, fit_intercept=True)
ridge.fit()
y_plot_ridge = ridge.predict(x_plot)

plt.plot(x_plot[:, 0], y_plot_ridge[:, 0], label='MQO regularizado (λ=0.25)', linewidth=2)

# Modelo da média
modelo_media = MeanModel(y)
modelo_media.fit()
y_plot_media = modelo_media.predict(x_plot)

plt.plot(x_plot[:, 0], y_plot_media[:, 0], label='Modelo da média', linewidth=2)

plt.legend()
plt.grid(True)
plt.show()

# Histograma dos resíduos do MQO tradicional
plt.figure(figsize=(8, 5))
epsilon = y - lr.predict(X)
plt.hist(epsilon, edgecolor='k')
plt.title(f'Resíduos do MQO tradicional | Média: {np.mean(epsilon):.4f}')
plt.xlabel("Erro")
plt.ylabel("Frequência")
plt.grid(True)
plt.show()