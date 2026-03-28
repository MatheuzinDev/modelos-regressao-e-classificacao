import numpy as np
import matplotlib.pyplot as plt
from modelos import LinearRegression, MeanModel, RidgeRegression


# Config
ARQUIVO_DADOS = "aerogerador.dat"
R = 500
LAMBDAS = [0, 0.25, 0.5, 0.75, 1]
SEED = 42

# Se True, salva os gráficos em PNG
SALVAR_FIGURAS = False


# Métricas
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def r2_score(y_true, y_pred):
    sse = np.sum((y_true - y_pred) ** 2)
    y_bar = np.mean(y_true)
    sst = np.sum((y_true - y_bar) ** 2)
    return 1 - (sse / sst)


def resumo_estatistico(valores):
    valores = np.asarray(valores)
    return {
        "media": np.mean(valores),
        "desvio_padrao": np.std(valores),
        "maior": np.max(valores),
        "menor": np.min(valores),
    }


def salvar_ou_mostrar(nome_arquivo=None):
    if SALVAR_FIGURAS and nome_arquivo:
        plt.tight_layout()
        plt.savefig(nome_arquivo, dpi=300, bbox_inches="tight")
    plt.show()


# Dados
np.random.seed(SEED)

data = np.loadtxt(ARQUIVO_DADOS)
X = data[:, 0:1]
y = data[:, 1:2]

x_plot = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)


# Plot - Espalhamento dos dados
plt.figure(figsize=(8, 5))
plt.scatter(X[:, 0], y[:, 0], alpha=0.7, edgecolor="k", label="Dados observados")
plt.xlabel("Velocidade do vento")
plt.ylabel("Potência gerada")
plt.title("Gráfico de espalhamento - Aerogerador")
plt.legend()
plt.grid(True)
salvar_ou_mostrar("regressao_01_espalhamento.png")


# Ajuste dos modelos
modelo_media = MeanModel(y)
modelo_media.fit()
y_plot_media = modelo_media.predict(x_plot)

modelo_mqo = LinearRegression(X, y, fit_intercept=True)
modelo_mqo.fit()
y_plot_mqo = modelo_mqo.predict(x_plot)

predicoes_ridge_plot = {}
for lamb in LAMBDAS:
    ridge = RidgeRegression(X, y, lamb=lamb, fit_intercept=True)
    ridge.fit()
    predicoes_ridge_plot[lamb] = ridge.predict(x_plot)

plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], y[:, 0], alpha=0.45, edgecolor="k", label="Dados observados")
plt.plot(x_plot[:, 0], y_plot_media[:, 0], linewidth=2, label="Modelo da média")
plt.plot(x_plot[:, 0], y_plot_mqo[:, 0], linewidth=2, label="MQO tradicional")

for lamb in LAMBDAS:
    plt.plot(
        x_plot[:, 0],
        predicoes_ridge_plot[lamb][:, 0],
        linewidth=1.8,
        linestyle="--",
        label=f"Ridge (λ={lamb})"
    )

plt.xlabel("Velocidade do vento")
plt.ylabel("Potência gerada")
plt.title("Comparação dos modelos de regressão")
plt.legend()
plt.grid(True)
salvar_ou_mostrar("regressao_02_comparacao_modelos.png")

# RANDOM SUBSAMPLING VALIDATION (R = 500)
resultados = {
    "Media": {"MSE": [], "R2": []},
    "MQO": {"MSE": [], "R2": []},
}

for lamb in LAMBDAS:
    resultados[f"Ridge_{lamb}"] = {"MSE": [], "R2": []}

N = X.shape[0]

for _ in range(R):
    idx = np.random.permutation(N)
    X_emb = X[idx, :]
    y_emb = y[idx, :]

    corte = int(N * 0.8)
    X_treino, X_teste = X_emb[:corte, :], X_emb[corte:, :]
    y_treino, y_teste = y_emb[:corte, :], y_emb[corte:, :]

    # Modelo da média
    media_model = MeanModel(y_treino)
    media_model.fit()
    y_hat = media_model.predict(X_teste)
    resultados["Media"]["MSE"].append(mse(y_teste, y_hat))
    resultados["Media"]["R2"].append(r2_score(y_teste, y_hat))

    # MQO
    ols = LinearRegression(X_treino, y_treino, fit_intercept=True)
    ols.fit()
    y_hat = ols.predict(X_teste)
    resultados["MQO"]["MSE"].append(mse(y_teste, y_hat))
    resultados["MQO"]["R2"].append(r2_score(y_teste, y_hat))

    # Ridge
    for lamb in LAMBDAS:
        nome = f"Ridge_{lamb}"
        ridge = RidgeRegression(X_treino, y_treino, lamb=lamb, fit_intercept=True)
        ridge.fit()
        y_hat = ridge.predict(X_teste)
        resultados[nome]["MSE"].append(mse(y_teste, y_hat))
        resultados[nome]["R2"].append(r2_score(y_teste, y_hat))


# Plot - MSE
nomes_modelos = list(resultados.keys())
mse_dados = [resultados[nome]["MSE"] for nome in nomes_modelos]
r2_dados = [resultados[nome]["R2"] for nome in nomes_modelos]

plt.figure(figsize=(12, 6))
plt.boxplot(mse_dados, tick_labels=nomes_modelos)
plt.xticks(rotation=20)
plt.ylabel("MSE")
plt.title("Distribuição do MSE por modelo (500 rodadas)")
plt.grid(True, axis="y")
salvar_ou_mostrar("regressao_03_boxplot_mse.png")


# Plot - R²
plt.figure(figsize=(12, 6))
plt.boxplot(r2_dados, tick_labels=nomes_modelos)
plt.xticks(rotation=20)
plt.ylabel("R²")
plt.title("Distribuição do R² por modelo (500 rodadas)")
plt.grid(True, axis="y")
salvar_ou_mostrar("regressao_04_boxplot_r2.png")

# Resumo com as médias das métricas
medias_mse = [np.mean(resultados[nome]["MSE"]) for nome in nomes_modelos]
medias_r2 = [np.mean(resultados[nome]["R2"]) for nome in nomes_modelos]

plt.figure(figsize=(12, 6))
plt.bar(nomes_modelos, medias_mse)
plt.xticks(rotation=20)
plt.ylabel("MSE médio")
plt.title("Comparação do MSE médio entre os modelos")
plt.grid(True, axis="y")
salvar_ou_mostrar("regressao_05_media_mse.png")

plt.figure(figsize=(12, 6))
plt.bar(nomes_modelos, medias_r2)
plt.xticks(rotation=20)
plt.ylabel("R² médio")
plt.title("Comparação do R² médio entre os modelos")
plt.grid(True, axis="y")
salvar_ou_mostrar("regressao_06_media_r2.png")

# Histograma dos resíduos do MQO tradicional
modelo_mqo_residuos = LinearRegression(X, y, fit_intercept=True)
modelo_mqo_residuos.fit()
residuos = y - modelo_mqo_residuos.predict(X)

plt.figure(figsize=(8, 5))
plt.hist(residuos[:, 0], bins=30, edgecolor="k")
plt.xlabel("Resíduo")
plt.ylabel("Frequência")
plt.title(f"Histograma dos resíduos do MQO | Média = {np.mean(residuos):.4f}")
plt.grid(True)
salvar_ou_mostrar("regressao_07_histograma_residuos.png")

# Print das estatísticas finais
print("\n========== ESTATÍSTICAS FINAIS ==========")
for modelo, metricas in resultados.items():
    print(f"\nModelo: {modelo}")

    estat_mse = resumo_estatistico(metricas["MSE"])
    estat_r2 = resumo_estatistico(metricas["R2"])

    print("MSE")
    print(f"  Média         = {estat_mse['media']:.6f}")
    print(f"  Desvio-padrão = {estat_mse['desvio_padrao']:.6f}")
    print(f"  Maior valor   = {estat_mse['maior']:.6f}")
    print(f"  Menor valor   = {estat_mse['menor']:.6f}")

    print("R²")
    print(f"  Média         = {estat_r2['media']:.6f}")
    print(f"  Desvio-padrão = {estat_r2['desvio_padrao']:.6f}")
    print(f"  Maior valor   = {estat_r2['maior']:.6f}")
    print(f"  Menor valor   = {estat_r2['menor']:.6f}")
