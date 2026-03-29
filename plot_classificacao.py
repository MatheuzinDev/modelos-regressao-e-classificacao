import numpy as np
import matplotlib.pyplot as plt

# Carregar dados
data_emg = np.loadtxt("EMGsDataset.csv", delimiter=',')
X = data_emg[0:2, :]
y = data_emg[2, :]

plt.figure(figsize=(10, 6))
nomes_classes = ['Neutro', 'Sorriso', 'Sobrancelhas levantadas', 'Surpreso', 'Rabugento']
cores = ['blue', 'green', 'red', 'purple', 'orange']

for i in range(1, 6):
    idx = np.where(y == i)[0]
    plt.scatter(X[0, idx], X[1, idx], label=nomes_classes[i-1], color=cores[i-1], 
                alpha=1, s=20, edgecolor='black', linewidth=0.5)

plt.xlabel('Sensor 1 (Corrugador do Supercílio)')
plt.ylabel('Sensor 2 (Zigomático Maior)')
plt.title('Espalhamento - Sinais EMG por Expressão Facial')
plt.legend()
plt.grid(True)
plt.show()