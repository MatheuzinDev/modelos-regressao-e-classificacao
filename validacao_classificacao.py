import numpy as np
import time
from modelos import (
    MQOClassificacao,
    ClassificadorGaussianoTradicional,
    ClassificadorGaussianoCovGlobal,
    ClassificadorGaussianoCovAgregada,
    ClassificadorNaiveBayes,
    ClassificadorGaussianoRegularizado
)

print("A carregar os dados...")
data_emg = np.loadtxt("EMGsDataset.csv", delimiter=',')
X_raw = data_emg[0:2, :] 
y_raw = data_emg[2, :]    

N = X_raw.shape[1]
C = 5

Y_mqo = np.zeros((N, C))
for i in range(N):
    Y_mqo[i, int(y_raw[i]) - 1] = 1


# K-FOLD CROSS VALIDATION PARA O MODELO DE FRIEDMAN
lambdas_para_testar = [0, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
k_folds = 5
tamanho_fold = N // k_folds

idx_kfold = np.random.permutation(N)

melhor_lamb = 0
melhor_acc_kfold = -1

print("\n--- A INICIAR K-FOLD (5 Folds) PARA O HIPERPARÂMETRO DE FRIEDMAN ---")
for lmb in lambdas_para_testar:
    acc_lista = []
    
    for i in range(k_folds):
        idx_val = idx_kfold[i * tamanho_fold : (i + 1) * tamanho_fold]
        idx_treino = np.concatenate([idx_kfold[:i * tamanho_fold], idx_kfold[(i + 1) * tamanho_fold:]])
        
        X_tr = X_raw[:, idx_treino]
        Y_tr = Y_mqo[idx_treino, :].T 
        X_val = X_raw[:, idx_val]
        Y_val_real = y_raw[idx_val]
        
        # Treino e Predição
        modelo = ClassificadorGaussianoRegularizado(lamb=lmb)
        modelo.fit(X_tr, Y_tr)
        preds = modelo.predict(X_val)
        
        # Acurácia
        acc = np.mean(preds == Y_val_real)
        acc_lista.append(acc)
        
    media_acc = np.mean(acc_lista)
    print(f"Lambda {lmb}: Acurácia média = {media_acc:.4f}")
    
    if media_acc > melhor_acc_kfold:
        melhor_acc_kfold = media_acc
        melhor_lamb = lmb

print(f">>> O melhor lambda encontrado foi: {melhor_lamb} (Acurácia: {melhor_acc_kfold:.4f}) <<<\n")


# SIMULAÇÃO DE MONTE CARLO (R=500) PARA TODOS OS MODELOS
R = 500
corte_treino = int(N * 0.8)

resultados = {
    "MQO Tradicional": [],
    "Classificador Gaussiano Tradicional": [],
    "Classificador Gaussiano (Cov. Global)": [],
    "Classificador Gaussiano (Cov. Agregada)": [],
    "Classificador de Bayes Ingênuo (Naive Bayes)": [],
    f"Classificador Gaussiano Regularizado (λ={melhor_lamb})": []
}

print(f"--- A INICIAR SIMULAÇÃO DE MONTE CARLO (R={R}) ---")
print("Essa etapa pode levar alguns minutos. Processando...")

tempo_inicio = time.time()

for r in range(R):
    idx = np.random.permutation(N)
    idx_treino = idx[:corte_treino]
    idx_teste = idx[corte_treino:]
    
    # DADOS PARA MQO
    X_mqo_tr = X_raw[:, idx_treino].T  
    Y_mqo_tr = Y_mqo[idx_treino, :]    
    X_mqo_ts = X_raw[:, idx_teste].T
    
    # DADOS PARA GAUSSIANOS
    X_g_tr = X_raw[:, idx_treino]      
    Y_g_tr = Y_mqo[idx_treino, :].T    
    X_g_ts = X_raw[:, idx_teste]
    
    # Classes reais de teste (para calcular acurácia de todos)
    Y_real_ts = y_raw[idx_teste]
    
    # MQO
    mqo = MQOClassificacao(X_mqo_tr, Y_mqo_tr)
    mqo.fit()
    resultados["MQO Tradicional"].append(np.mean(mqo.predict(X_mqo_ts) == Y_real_ts))
    
    # Gaussiano Tradicional
    gt = ClassificadorGaussianoTradicional()
    gt.fit(X_g_tr, Y_g_tr)
    resultados["Classificador Gaussiano Tradicional"].append(np.mean(gt.predict(X_g_ts) == Y_real_ts))
    
    # Gaussiano Cov Global
    gg = ClassificadorGaussianoCovGlobal()
    gg.fit(X_g_tr, Y_g_tr)
    resultados["Classificador Gaussiano (Cov. Global)"].append(np.mean(gg.predict(X_g_ts) == Y_real_ts))
    
    # Gaussiano Cov Agregada
    ga = ClassificadorGaussianoCovAgregada()
    ga.fit(X_g_tr, Y_g_tr)
    resultados["Classificador Gaussiano (Cov. Agregada)"].append(np.mean(ga.predict(X_g_ts) == Y_real_ts))
    
    # Naive Bayes
    nb = ClassificadorNaiveBayes()
    nb.fit(X_g_tr, Y_g_tr)
    resultados["Classificador de Bayes Ingênuo (Naive Bayes)"].append(np.mean(nb.predict(X_g_ts) == Y_real_ts))
    
    # Friedman (com o melhor lambda)
    gf = ClassificadorGaussianoRegularizado(lamb=melhor_lamb)
    gf.fit(X_g_tr, Y_g_tr)
    resultados[f"Classificador Gaussiano Regularizado (λ={melhor_lamb})"].append(np.mean(gf.predict(X_g_ts) == Y_real_ts))
    
    if (r + 1) % 50 == 0:
        print(f"Rodada {r+1}/{R} concluída...")

tempo_fim = time.time()
print(f"\nSimulação concluída em {(tempo_fim - tempo_inicio)/60:.2f} minutos.\n")

# IMPRESSÃO DOS RESULTADOS FINAIS
print("-" * 70)
print(f"{'Modelo':<50} | {'Média':<7} | {'Desvio':<7} | {'Maior':<7} | {'Menor':<7}")
print("-" * 70)

for modelo, acc_lista in resultados.items():
    acc_arr = np.array(acc_lista)
    media = np.mean(acc_arr)
    desvio = np.std(acc_arr)
    maior = np.max(acc_arr)
    menor = np.min(acc_arr)
    
    print(f"{modelo:<50} | {media:.4f}  | {desvio:.4f}  | {maior:.4f}  | {menor:.4f}")

print("-" * 70)