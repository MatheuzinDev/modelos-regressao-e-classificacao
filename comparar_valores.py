import numpy as np
from modelos import LinearRegression, MeanModel, RidgeRegression

# Carregar os dados
data = np.loadtxt("aerogerador.dat")

X = data[:, 0:1]
y = data[:, 1:2]
N, p = X.shape

R = 500
lambdas = [0, 0.25, 0.5, 0.75, 1]

resultados = {
    "Media": {"MSE": [], "R2": []},
    "MQO": {"MSE": [], "R2": []},
}

for lmb in lambdas:
    nome = f"Ridge_{lmb}"
    resultados[nome] = {"MSE": [], "R2": []}

for r in range(R):
    # Embaralhar os dados
    idx = np.random.permutation(N)
    X_aleatorio = X[idx, :]
    y_aleatorio = y[idx, :]

    # Divisão 80/20
    corte = int(N * 0.8)

    X_treino = X_aleatorio[:corte, :]
    y_treino = y_aleatorio[:corte, :]

    X_teste = X_aleatorio[corte:, :]
    y_teste = y_aleatorio[corte:, :]

    # Modelo da média
    media = MeanModel(y_treino)
    media.fit()
    y_hat = media.predict(X_teste)

    mse = np.mean((y_teste - y_hat) ** 2)
    sse = np.sum((y_teste - y_hat) ** 2)
    y_bar = np.mean(y_teste)
    sst = np.sum((y_teste - y_bar) ** 2)
    r2 = 1 - (sse / sst)

    resultados["Media"]["MSE"].append(mse)
    resultados["Media"]["R2"].append(r2)

    # MQO tradicional
    ols = LinearRegression(X_treino, y_treino, fit_intercept=True)
    ols.fit()
    y_hat = ols.predict(X_teste)

    mse = np.mean((y_teste - y_hat) ** 2)
    sse = np.sum((y_teste - y_hat) ** 2)
    y_bar = np.mean(y_teste)
    sst = np.sum((y_teste - y_bar) ** 2)
    r2 = 1 - (sse / sst)

    resultados["MQO"]["MSE"].append(mse)
    resultados["MQO"]["R2"].append(r2)

    # MQO regularizado
    for lmb in lambdas:
        nome = f"Ridge_{lmb}"

        ridge = RidgeRegression(X_treino, y_treino, lamb=lmb, fit_intercept=True)
        ridge.fit()
        y_hat = ridge.predict(X_teste)

        mse = np.mean((y_teste - y_hat) ** 2)
        sse = np.sum((y_teste - y_hat) ** 2)
        y_bar = np.mean(y_teste)
        sst = np.sum((y_teste - y_bar) ** 2)
        r2 = 1 - (sse / sst)

        resultados[nome]["MSE"].append(mse)
        resultados[nome]["R2"].append(r2)

# Mostrar estatísticas finais
for modelo, metricas in resultados.items():
    print(f"\nModelo: {modelo}")

    mse_vals = np.array(metricas["MSE"])
    r2_vals = np.array(metricas["R2"])

    print("MSE")
    print(f"  Média         = {np.mean(mse_vals):.6f}")
    print(f"  Desvio-padrão = {np.std(mse_vals):.6f}")
    print(f"  Maior valor   = {np.max(mse_vals):.6f}")
    print(f"  Menor valor   = {np.min(mse_vals):.6f}")

    print("R²")
    print(f"  Média         = {np.mean(r2_vals):.6f}")
    print(f"  Desvio-padrão = {np.std(r2_vals):.6f}")
    print(f"  Maior valor   = {np.max(r2_vals):.6f}")
    print(f"  Menor valor   = {np.min(r2_vals):.6f}")