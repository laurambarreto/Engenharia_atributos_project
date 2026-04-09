## -- IMPORT DAS BIBLIOTECAS NECESSÁRIAS -- ##
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, colormaps
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from sklearn.cluster import DBSCAN
import seaborn as sns
from scipy.stats import kstest, zscore
from scipy.stats import kruskal
from scipy.stats import mannwhitneyu
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import welch
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from skrebate import ReliefF
from sklearn.feature_selection import f_classif  
import matplotlib.patches as mpatches

## ------------ EXERCÍCIO 2 ------------ ##
## -- DADOS DE TODOS OS PARTICIPANTES NUMA MATRIZ -- ##
def matriz ():
    array_individuo = []
    for part in range(0, 15):
        for device in range(1,6):
            df = pd.read_csv(f"Part {part}/part{part}dev{device}.csv")
            df["Part"] = int(part+1)
            array_individuo.append (df.values)
        matriz = np.vstack(array_individuo) 
    return matriz

dados = matriz ()

# Verificar a quantidade de atividades distintas
atividades = np.unique(dados[:, 11].astype(int))
print(atividades)

## ------------ EXERCÍCIO 3 ------------ ##
## -- MÓDULOS DAS MEDIÇÕES DOS 3 SENSORES -- ##
def modulo ():
    # Cria lista com os módulos
    novas_linhas = []
    for i in range (dados.shape [0]):
        # Cálculo dos módulos (√(sensor_x^2 + sensor_y^2 + sensor_z^2))
        modulo_acc = math.sqrt (float (dados[i][1])**2 + float (dados[i][2])**2 + float (dados[i][3])**2)
        modulo_gyr = math.sqrt (float (dados[i][4])**2 + float (dados[i][5])**2 + float (dados[i][6])**2)
        modulo_mag = math.sqrt (float (dados[i][7])**2 + float (dados[i][8])**2 + float (dados[i][9])**2)

        # Adiciona os módulos no final da linha
        nova_linha = np.append(dados[i], [modulo_acc, modulo_gyr, modulo_mag])

        # Adiciona nova linha à lista novas_linhas
        novas_linhas.append(nova_linha)

    # Converter a lista de linhas de volta para array numpy
    return np.array(novas_linhas)

dados = modulo ()

## ------------ EXERCÍCIO 3.1 ------------ ##
## -- FUNÇÃO QUE CRIA BOXPLOTS -- ##
# Todas as atividades (eixo x)
# Um gráfico para cada device
# Cada gráfico tem 3 subplots para cada um dos módulos
def boxplots():
    nomes_modulos = ['Acelerómetro', 'Giroscópio', 'Magnetómetro'] # Nomes das colunas
    colunas_modulos = [13, 14, 15] # Índices dos módulos

    # Uma figura para cada device
    for device in range (1, 6): 
        
        # Filtrar só os dados do device
        device_data = dados[dados[:, 0] == device]

        # Criar uma figura com 3 subplots lado a lado
        plt.figure(figsize = (12, 5))
        plt.suptitle(f'Device {device} - Boxplots por Módulos e Atividades', fontsize = 22, fontname = 'Trebuchet MS', fontweight = 'bold', color = '#c00000')

        # Para cada um dos 3 módulos
        for i, (nome_modulo, col_modulo) in enumerate(zip(nomes_modulos, colunas_modulos), start = 1):
            # Criar listas de valores por atividade (para este módulo)
            valores_boxplot = [device_data[device_data[:, 11] == a, col_modulo] for a in atividades]

            # Subplot desse módulo com as 16 atividades
            plt.subplot(1, 3, i)

            # Estilo dos outliers
            flierprops = dict(
                marker = 'o', # Forma do ponto
                markerfacecolor = "#f41313", # Cor do ponto
                markersize = 2, # Tamanho
                alpha = 0.5 # Transparência
            )

            # Estilo da linha da mediana
            medianprops = dict(
                color = '#f41313', # Cor da mediana
                linewidth = 2 # Espessura da linha
            )

            plt.boxplot(valores_boxplot, showfliers = True, flierprops = flierprops, medianprops = medianprops)
            plt.title(nome_modulo, fontname = 'Trebuchet MS', fontsize = 16, fontweight = 'bold')
            plt.xlabel('Atividade')
            plt.ylabel('Módulo')
            # Muda a rotação dos valores das atividades no eixo x, para melhor visualização
            plt.xticks(range(1, len(atividades) + 1), atividades, rotation = 45)

            # Grelhas a tracejado e ligeiramente transparentes
            plt.grid(True, linestyle = '--', alpha = 0.6)

        plt.tight_layout(rect = [0, 0, 1, 0.93]) # Espaçamento entre subplots e margens para não haver sobreposição
        plt.show()

boxplots()

## ------------ EXERCÍCIO 3.2 ------------ ##
## -- DENSIDADE DE OUTLIERS NO DEVICE 2 -- ##
# Todas as atividades (eixo x)
# Apenas para o device 2
# O gráfico tem 3 subplots para cada um dos módulos
def densidade_outliers_IQR_device2 ():
    nomes_modulos = ['Acelerómetro', 'Giroscópio', 'Magnetómetro'] # Nomes dos módulos
    colunas_modulos = [13, 14, 15] # Índices das colunas dos módulos

    # Filtrar só os dados do device 2
    device_data = dados[dados[:, 0] == 2]

    # Guardar as densidades para o plot
    densidades_por_modulo = []

    for (nome_modulo, coluna_modulo) in zip(nomes_modulos, colunas_modulos):
        print(f'\nDensidade de Outliers por atividade - {nome_modulo} (Device 2)')

        densidades = []
        for atividade in atividades:
            # Filtrar os dados para a atividade atual
            atividade_data = device_data[device_data[:, 11] == atividade, coluna_modulo].astype(float)

            # Calcular Q1, Q3 e IQR
            Q1 = np.percentile(atividade_data, 25)
            Q3 = np.percentile(atividade_data, 75)

            # Calcular IQR
            IQR = Q3 - Q1

            # Definir limites para outliers
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Contar outliers
            outliers = atividade_data[(atividade_data < lower_bound) | (atividade_data > upper_bound)]
            num_outliers = len(outliers)
            total_points = len(atividade_data)

            # Calcular densidade (%)
            densidade_outlier = (num_outliers / total_points * 100) if total_points > 0 else 0
            densidades.append(densidade_outlier)

            print(f'Atividade {atividade}: {densidade_outlier:.2f}%')

        densidades_por_modulo.append(densidades)


    fig, axs = plt.subplots(1, 3, figsize = (14, 5), sharey = True) # 3 gráficos lado a lado

    for i, ax in enumerate(axs):
        ax.set_axisbelow(True) # Grids desenhadas por trás das barras
        ax.grid(True, linestyle = '--', linewidth = 0.7, alpha = 0.4) 
        # which = 'major' -> grelha apenas nas linhas de ticks principais
        # alpha = transparência
        ax.bar(range(len(atividades)), densidades_por_modulo[i], color = '#FFC222', edgecolor = '#c00000', linewidth = 0.5)
        ax.set_ylim(0, 25) 
        ax.set_title(nomes_modulos[i], fontname = 'Trebuchet MS', fontsize = 16, fontweight = 'bold')
        ax.set_xlabel('Atividade')

        if i == 0: # Se for o primeiro subplot, para não haver repetições
            ax.set_ylabel('Densidade de Outliers (%)') # Definimos a label do eixo y

        ax.set_xticks(range(len(atividades))) # Atividades no eixo x
        # Muda a rotação dos valores das atividades no eixo x, para melhor visualização
        ax.set_xticklabels(atividades, rotation = 45) 

    fig.suptitle('Densidade de Outliers por Módulo e Atividade (Device 2, IQR)', fontsize = 22, fontweight = 'bold', fontname = 'Trebuchet MS', color = '#c00000')
    plt.tight_layout(rect = [0, 0, 1, 0.95]) # Espaçamento entre subplots e margens para não haver sobreposição
    plt.show()

densidade_outliers_IQR_device2()

## ------------ EXERCÍCIO 3.3 ------------ ##
# -- DETEÇÃO DE OUTLIERS ATRAVÉS DO Z-SCORE -- ##
def zscore_outliers(array, k):
    # Calcular média e desvio padrão
    mean = np.mean(array)
    std = np.std(array)

    # Calcular Z-scores
    zscores = np.abs((array - mean) / std)

    # Índices dos outliers
    indices_outliers = np.where(zscores > k)[0]

    return indices_outliers


## ------------ EXERCÍCIO 3.4 ------------ ##
## -- GRÁFICOS COM OS OUTLIERS IDENTIFICADOS ATRAVÉS DE Z-SCORE POR DEVICE -- ##
# Todas as atividades (eixo x)
# Apenas para 1 device
# Cada gráfico tem 3 subplots para cada um dos módulos
def plot_outliers_por_device(k, device_id):
    nomes_modulos = ['Acelerómetro', 'Giroscópio', 'Magnetómetro'] # Nomes dos módulos
    colunas_modulos = [13, 14, 15] # Índices das colunas dos módulos
   
    # Filtrar só os dados do device
    device_data = dados[dados[:, 0] == device_id]

    # Criar uma figura com 3 subplots lado a lado
    plt.figure(figsize = (15, 6))
    plt.suptitle(f'Device {device_id} - Outliers (Z-score > {k})', fontsize = 22, fontweight = 'bold', fontname = 'Trebuchet MS', color = '#c00000')

    # Para cada um dos 3 módulos
    for subplot_idx, (nome_modulo, coluna_modulo) in enumerate(zip(nomes_modulos, colunas_modulos), start = 1):
        plt.subplot(1, 3, subplot_idx)

        plt.grid(True, linestyle = '--', alpha = 0.6, zorder = 0) # zorder = 0 -> primeira camada, fica por trás de tudo

        for atividade in atividades:
            # Filtrar dados da atividade
            atividade_data = device_data[device_data[:, 11].astype(int) == atividade, coluna_modulo].astype(float)

            # Identificar outliers na atividade
            indices_outliers = zscore_outliers(atividade_data, k)

            # Posições no eixo X (para separar visualmente as atividades)
            x_vals = np.full(len(atividade_data), atividade)

            # Pontos normais (azuis)
            plt.scatter(x_vals, atividade_data, color = 'blue', s = 7, alpha = 0.6, zorder = 2) # zorder = 2 -> segunda camada

            # Outliers (vermelhos)
            if len(indices_outliers) > 0:
                plt.scatter(x_vals[indices_outliers], atividade_data[indices_outliers], color = 'red', s = 10, label = 'Outlier', zorder = 3) # zorder = 3 -> fica à frente de tudo

        plt.title(nome_modulo, fontname = 'Trebuchet MS', fontsize = 16, fontweight = 'bold')
        plt.xlabel('Atividade')
        plt.ylabel('Módulo')
        plt.xticks(atividades)

    plt.tight_layout(rect = [0, 0, 1, 0.93]) # Espaçamento entre subplots e margens para não haver sobreposição
    plt.show()


plot_outliers_por_device(k = 3, device_id = 2)
plot_outliers_por_device(k = 3.5, device_id = 2)
plot_outliers_por_device(k = 4, device_id = 2)


## -- DENSIDADE DE OUTLIERS COM Z-SCORE (SÓ DEVICE 2) -- ##
def densidade_outliers_zscore_device2(k):
    nomes_modulos = ['Acelerómetro', 'Giroscópio', 'Magnetómetro'] # Nomes dos módulos
    colunas_modulos = [13, 14, 15]  # Índices dos módulos

    # Filtrar só os dados do device 2
    device_data = dados[dados[:, 0] == 2]

    # Guardar as densidades para o plot
    densidades_por_modulo = []

    for (nome_modulo, col_modulo) in zip(nomes_modulos, colunas_modulos):
        print(f'\nDensidade de Outliers por atividade - {nome_modulo} (Device 2, Z-score > {k})')

        densidades = []
        for atividade in atividades:
            # Filtrar os dados para a atividade atual
            atividade_data = device_data[device_data[:, 11] == atividade, col_modulo].astype(float)

            # Identificar outliers usando Z-score
            indices_outliers = zscore_outliers(atividade_data, k)
            num_outliers = len(indices_outliers)
            total_points = len(atividade_data)

            # Calcular densidade (%)
            densidade_outlier = (num_outliers / total_points * 100) if total_points > 0 else 0
            densidades.append(densidade_outlier)

            print(f'Atividade {atividade}: {densidade_outlier:.2f}%')

        densidades_por_modulo.append(densidades)

    # Plot das densidades
    fig, axs = plt.subplots(1, 3, figsize = (14, 5), sharey = True)

    for i, ax in enumerate(axs):
        ax.set_axisbelow(True) # Grids ficam por trás das barras
        ax.grid(True, linestyle = '--', linewidth = 0.7, alpha = 0.4)
        ax.bar(range(len(atividades)), densidades_por_modulo[i], color = '#FFC222', edgecolor = '#c00000', linewidth = 0.5)
        ax.set_ylim(0, 25)
        ax.set_title(nomes_modulos[i], fontname = 'Trebuchet MS', fontsize = 16, fontweight = 'bold')
        ax.set_xlabel('Atividade')

        # Se for o primeiro subplot, para não haver repetições
        if i == 0:
            ax.set_ylabel('Densidade de Outliers (%)') # Definimos a label do eixo y

        ax.set_xticks(range(len(atividades)))
        ax.set_xticklabels(atividades, rotation = 45)

    fig.suptitle(f'Densidade de Outliers por Módulo e Atividade (Device 2, Z-score > {k})', fontsize = 22, fontweight = 'bold', fontname = 'Trebuchet MS', color = '#c00000')
    plt.tight_layout(rect = [0, 0, 1, 0.95]) # Espaçamento entre subplots e margens para não haver sobreposição
    plt.show()


densidade_outliers_zscore_device2 (k = 3)
densidade_outliers_zscore_device2 (k = 3.5)
densidade_outliers_zscore_device2 (k = 4)

## ------------ EXERCÍCIO 3.6 ------------ ##
## -- K-MEANS QUE AGRUPA DADOS EM CLUSTERS E RETORNA OS ÍNDICES DOS OUTLIERS -- ##
def kmeans (dados, n, max_iter = 100):
    # Escolher n centroides iniciais aleatoriamente do conjunto de dados
    indices = np.random.choice (dados.shape[0], n, replace = False)
    centroides = dados [indices]

    for _ in range (max_iter):
        # Calcular a distância de cada ponto a todos os centroides
        distancias = np.linalg.norm (dados [:, np.newaxis] - centroides, axis = 2)
        # Atribuir cada ponto ao centroide mais próximo
        labels = np.argmin (distancias, axis = 1)
        # Guardar centroides da iteração anterior
        centroides_old = centroides.copy ()
        # Atualizar os centroides como a média dos pontos atribuídos a cada cluster
        for i in range (n):
            pontos_cluster = dados [labels == i]
            centroides [i] = np.mean (pontos_cluster, axis = 0)
        # Se os centroides não mudaram significativamente, parar
        if np.allclose (centroides, centroides_old):
            break
    
    outliers_idx = []
    for i in range(n):
        pontos_cluster = dados[labels == i]
        centroid = centroides[i]

        # Distâncias dos pontos ao respetivo centróide
        distancias_cluster = np.linalg.norm(pontos_cluster - centroid, axis = 1)

        # Calcular Q1, Q3 e IQR
        Q1 = np.percentile(distancias_cluster, 25)
        Q3 = np.percentile(distancias_cluster, 75)
        IQR = Q3 - Q1

        # Limites para outliers
        lower = Q1 - 2 * IQR
        upper = Q3 + 2 * IQR

        # Índices dos outliers
        cluster_indices = np.where(labels == i)[0]
        outlier_indices = cluster_indices[(distancias_cluster < lower) | (distancias_cluster > upper)]
        outliers_idx.extend(outlier_indices)

    return centroides, labels, np.array(outliers_idx)


## -- FUNÇÃO AUXILIAR: CLAREAR COR -- ##
def clarear_cor(cor, fator = 1.6):
    # Clareia uma cor misturando-a com branco
    # fator > 1 -> mais clara
    cor_rgb = np.array(mcolors.to_rgb(cor))
    cor_clara = 1 - (1 - cor_rgb) / fator
    return np.clip(cor_clara, 0, 1)


## -- K-MEANS PARA IDENTIFICAR OUTLIERS POR DEVICE -- ##
# Apenas para um device
# Para todas as atividades
def kmeans_outliers_3D_apenasPorDevice(device_id, n_clusters):
    # Filtrar só os dados deste device
    device_data = dados[dados[:, 0] == device_id]
    device_modulos = device_data[:, 13:16].astype(float)

    # Normalizar os módulos (z-score)
    mean = np.mean(device_modulos, axis = 0)
    std = np.std(device_modulos, axis = 0)
    device_modulos_norm = (device_modulos - mean) / std

    # Confirmar normalização
    print("\nApós normalização (z-score):")
    print("Médias:", np.mean(device_modulos_norm, axis = 0))
    print("Desvios padrão: ", np.std (device_modulos_norm, axis = 0))
    print("Mínimos normalizados:", np.min(device_modulos_norm, axis = 0))
    print("Máximos normalizados:", np.max(device_modulos_norm, axis = 0))

    # Contagem de pontos <= 0 e > 0
    for i, nome in enumerate(['Acelerómetro', 'Giroscópio', 'Magnetómetro']):
        n_menor_igual_0 = np.sum(device_modulos_norm[:, i] <= 0)
        n_maior_0 = np.sum(device_modulos_norm[:, i] > 0)
        print(f"\n{nome}:")
        print(f"  Pontos ≤ 0: {n_menor_igual_0}")
        print(f"  Pontos > 0: {n_maior_0}")

    # Aplicar K-means 
    centroides, labels, indice_outliers = kmeans(device_modulos_norm, n_clusters)

    # Paleta de cores fortes
    cores_base = list(mcolors.TABLEAU_COLORS.values())

    # Garantir que todos os clusters têm cor diferente
    if n_clusters > len(cores_base):
        cores_base = list(mcolors.CSS4_COLORS.values())[:n_clusters]

    # -- Plot 3D dos clusters -- #
    fig = plt.figure(figsize = (10, 7))
    ax = fig.add_subplot(111, projection = '3d')

    for i in range(n_clusters):
        cluster_mask = labels == i
        cor_base = cores_base[i % len(cores_base)]
        cor_clara = clarear_cor(cor_base, fator=2.0)

        # Pontos normais
        normais_idx = np.setdiff1d(np.where(cluster_mask)[0], indice_outliers)
        if len(normais_idx) > 0:
            ax.scatter(device_modulos_norm[normais_idx, 0],
                       device_modulos_norm[normais_idx, 1],
                       device_modulos_norm[normais_idx, 2],
                       c = [cor_clara], s = 2, alpha = 0.2, marker = 'o',
                       label = f'Cluster {i+1}')

        # Outliers
        outliers_cluster = np.intersect1d(np.where(cluster_mask)[0], indice_outliers)
        if len(outliers_cluster) > 0:
            ax.scatter(device_modulos_norm[outliers_cluster, 0],
                       device_modulos_norm[outliers_cluster, 1],
                       device_modulos_norm[outliers_cluster, 2],
                       c = [cor_base], s = 10, alpha = 0.7, marker = 'P',
                       label = f'Outliers C{i+1}')

    # Centróides
    ax.scatter(centroides[:, 0], centroides[:, 1], centroides[:, 2], c = 'black', s = 90, marker = 'P', label = 'Centroides')

    ax.set_title(f"Device {device_id} (k = {n_clusters})", fontsize = 22, fontweight = 'bold', fontname = 'Trebuchet MS', color = '#c00000')
    ax.set_xlabel('Módulo ACC')
    ax.set_ylabel('Módulo GYR')
    ax.set_zlabel('Módulo MAG')
    ax.legend(loc = 'upper right') # Legenda fica no canto superior direito
    plt.show()

    # -- Plot 2D -- #
    atividades = device_data[:, 11].astype(int)
    mod_acc = device_data[:, 13].astype(float)
    mod_gyr = device_data[:, 14].astype(float)
    mod_mag = device_data[:, 15].astype(float)

    mods = [mod_acc, mod_gyr, mod_mag]
    nomes = ['Acelerómetro', 'Giroscópio', 'Magnetómetro']

    labels_outliers = np.zeros(len(device_data), dtype = bool)
    labels_outliers[indice_outliers] = True

    fig, axes = plt.subplots(1, 3, figsize = (15, 5), sharey = False)

    for i in range(3):
        ax = axes[i]
        ax.grid(True, linestyle = '--', alpha = 0.6)
        # Não outliers
        ax.scatter(atividades[~labels_outliers], mods[i][~labels_outliers], color = 'blue', s = 10, alpha = 0.8, label = 'Normal')
        # Outliers
        ax.scatter(atividades[labels_outliers], mods[i][labels_outliers], color = 'red', s = 10, alpha = 0.2, label = 'Outlier')
        ax.set_xlabel('Atividade')
        plt.xticks(np.unique(dados[:, 11].astype(int)))
        ax.set_ylabel('Módulo ' + nomes[i])
        ax.set_title(nomes[i])

    axes[0].legend() # Colocar legenda apenas para o primeiro subplot
    fig.suptitle(f'Outliers por Atividade com o Kmeans — Device {device_id}',  fontsize = 22, fontweight = 'bold', fontname = 'Trebuchet MS', color = '#c00000')
    plt.show()

    # --- Cálculo de densidade de outliers por módulo ---
    densidades_por_modulo = []
    for i in range(3):
        total_por_atividade = np.array([np.sum(atividades == a) for a in np.unique(atividades)])
        outliers_por_atividade = np.array([np.sum((atividades == a) & labels_outliers) for a in np.unique(atividades)])
        densidade = (outliers_por_atividade / total_por_atividade * 100)
        densidades_por_modulo.append(densidade)

    # Plot das densidades de outliers (%)
    fig, axs = plt.subplots(1, 3, figsize = (14, 5), sharey = True)

    for i, ax in enumerate(axs):
        ax.set_axisbelow(True)
        ax.grid(True, which = 'major', linestyle = '--', linewidth = 0.7, alpha = 0.4)
        ax.bar(range(len(np.unique(atividades))), densidades_por_modulo[i], color = '#FFC222', edgecolor = '#c00000', linewidth = 0.5)
        ax.set_ylim(0, 25)
        ax.set_title(nomes[i], fontname = 'Trebuchet MS', fontsize = 16, fontweight = 'bold')
        ax.set_xlabel('Atividade')

        # Se for o primeiro subplot, para não haver repetições
        if i == 0:
            ax.set_ylabel('Densidade de Outliers (%)') # Definimos a label do eixo y

        ax.set_xticks(range(len(np.unique(atividades))))
        ax.set_xticklabels(np.unique(atividades), rotation = 45)

    fig.suptitle(f'Densidade de Outliers por Módulo e Atividade — (Device {device_id}, k-means)', fontsize = 22, fontweight = 'bold', fontname = 'Trebuchet MS', color = '#c00000')
    plt.tight_layout(rect = [0, 0, 1, 0.95])
    plt.show()

kmeans_outliers_3D_apenasPorDevice(device_id = 2, n_clusters = 16)
'''
kmeans_outliers_3D_apenasPorDevice(device_id = 1, n_clusters = 16)
kmeans_outliers_3D_apenasPorDevice(device_id = 3, n_clusters = 16)
kmeans_outliers_3D_apenasPorDevice(device_id = 4, n_clusters = 16)
kmeans_outliers_3D_apenasPorDevice(device_id = 5, n_clusters = 16)
'''

## ------------ EXERCÍCIO 3.7.1 ------------ ##
## -- DBSCAN PARA DESCOBRIR OUTLIERS POR DEVICE E NUMA ATIVIDADE -- ##
# Apenas para um device e uma atividade
def dbscan_outliers_3D(device_id, atividade, eps = 0.5, min_samples = 5):
    # Filtrar só os dados deste device e atividade
    device_activity_data = dados[(dados[:, 0] == device_id) & (dados[:, 11] == atividade)]

    # Extrair colunas dos módulos (X, Y, Z)
    device_modulos =  device_activity_data[:, 13:16].astype(float)

    # Normalizar (z-score)
    mean = np.mean(device_modulos, axis = 0)
    std = np.std(device_modulos, axis = 0)
    device_modulos_norm = (device_modulos - mean) / std

    # Aplicar DBSCAN
    dbscan = DBSCAN(eps = eps, min_samples = min_samples)
    labels = dbscan.fit_predict(device_modulos_norm)

    # Identificar outliers
    mask_inliers = labels != -1
    mask_outliers = labels == -1

    # Número de clusters (sem contar os outliers)
    unique_clusters = np.unique(labels[mask_inliers])
    n_clusters = len(unique_clusters)

    # Criar colormap viridis
    cmap = colormaps.get_cmap('viridis')
    # Gerar cores igualmente espaçadas no viridis
    colors = cmap(np.linspace(0, 1, n_clusters))

    fig = plt.figure(figsize = (8, 6))
    ax = fig.add_subplot(111, projection = '3d')

    # Plot de cada cluster com uma cor fixa
    for i, cluster_id in enumerate(unique_clusters):
        cluster_points = device_modulos_norm[labels == cluster_id]
        ax.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            cluster_points[:, 2],
            color = colors[i],
            s = 25,
            label = f'Cluster {cluster_id}'
        )

    # Outliers a vermelho
    if np.any(mask_outliers):
        ax.scatter(
            device_modulos_norm[mask_outliers, 0],
            device_modulos_norm[mask_outliers, 1],
            device_modulos_norm[mask_outliers, 2],
            c = 'red',
            s = 20,
            alpha = 0.5,
            label = 'Outliers'
        )

    # Rótulos e legenda
    ax.set_xlabel('Módulo ACC')
    ax.set_ylabel('Módulo GYR')
    ax.set_zlabel('Módulo MAG')
    ax.set_title(f'DBSCAN 3D — Device {device_id}, Atividade {atividade}', fontsize = 22, fontweight = 'bold', fontname = 'Trebuchet MS', color = '#c00000')
    ax.legend()
    plt.show()


dbscan_outliers_3D(device_id = 2, atividade = 8)


## ------------ EXERCÍCIO 4.1 ------------ ##
## -- VERIFICA A NORMALIDADE DOS MÓDULOS -- ##
# Apenas para um device
# Para todas as atividades
def heatmap_normalidade(device_id):
    # Filtrar dados do device
    device_data = dados[dados[:, 0] == device_id]
    
    # Extrair colunas relevantes
    atividades = device_data[:, 11].astype(int)
    mod_acc = device_data[:, 13].astype(float)
    mod_gyr = device_data[:, 14].astype(float)
    mod_mag = device_data[:, 15].astype(float)

    # Lista de módulos e nomes
    mods = [mod_acc, mod_gyr, mod_mag]
    nomes = ['ACC', 'GYR', 'MAG']
    atividades_unicas = np.unique(atividades)

    # Matriz de p-values [módulo, atividade]
    p_matrix = np.zeros((3, len(atividades_unicas)))

    for i, mod in enumerate(mods):
        # i = índice do módulo (0,1,2)
        # mod = array do módulo correspondente
        for j, act in enumerate(atividades_unicas):
            # j = índice da atividade
            # act = código da atividade atual
            dados_atividade = mod[atividades == act] # Filtra os valores do módulo só para esta atividade
            # Normalizar com z-score antes do teste
            dados_norm = zscore(dados_atividade)
            # Aplicar o teste de Kolmogorov–Smirnov
            stat, p_value = kstest(dados_norm, 'norm')
            # Guardar o p-value na posição (i, j) da matriz
            p_matrix[i, j] = p_value

    # Criar heatmap
    plt.figure(figsize = (12, 4))
    sns.heatmap(
        p_matrix, annot = True, fmt = ".3f",
        xticklabels = atividades_unicas, yticklabels = nomes,
        cmap = "coolwarm", cbar_kws = {'label': 'p-value (Kolmogorov-Smirnov)'},
        vmin = 0, vmax = 1,
        linewidths = 0.4, linecolor = 'black'
    )
    plt.axhline(1, color = 'black', linewidth = 0.5)
    plt.title(f'Normalidade (KS-Test) — Device {device_id}', fontsize = 22, fontweight = 'bold', fontname = 'Trebuchet MS', color = '#c00000')
    plt.xlabel('Atividade')
    plt.ylabel('Módulo')
    plt.tight_layout()
    plt.show()

heatmap_normalidade(device_id = 2)
'''
heatmap_normalidade(device_id = 1)
heatmap_normalidade(device_id = 3)
heatmap_normalidade(device_id = 4)
heatmap_normalidade(device_id = 5)
'''

## -- IDENTIFICA SE HÁ DIFERENÇAS ESTATISTICAMENTE SIGNIFICATIVAS NOS MÓDULOS ATRAVÉS DO TESTE DE KRUSKAL WALLIS -- ## 
def testar_significancia_kruskal(device_id):
    # Filtrar apenas os dados do device
    device_data = dados[dados[:, 0] == device_id]

    # Extrair colunas relevantes
    atividades = device_data[:, 11].astype(int)
    mod_acc = device_data[:, 13].astype(float)
    mod_gyr = device_data[:, 14].astype(float)
    mod_mag = device_data[:, 15].astype(float)

    # Nomes e módulos
    mods = [mod_acc, mod_gyr, mod_mag]
    nomes = ['ACC', 'GYR', 'MAG']
    atividades_unicas = np.unique(atividades)

    # Guardar p-values
    p_vals = []

    print(f"\nTeste de Kruskal-Wallis — Device {device_id}")
    print("---------------------------------------------------")

    for nome, mod in zip(nomes, mods):
        # Criar lista com os valores do módulo por atividade
        grupos = [mod[atividades == act] for act in atividades_unicas]

        # Aplicar o teste de Kruskal–Wallis
        stat, p_value = kruskal(*grupos) # Passa os grupos como argumentos separados
        p_vals.append(p_value)

        # Imprimir resultado e interpretação
        interpretacao = "Diferenças significativas" if p_value < 0.05 else "Sem diferenças significativas" # p-value < 0.05 -> Diferenças significativas, p-value > 0.05 -> Sem diferenças significativas
        print(f"{nome}: p = {p_value:.5f} → {interpretacao}")

testar_significancia_kruskal(device_id = 2)
'''
testar_significancia_kruskal(device_id = 1)
testar_significancia_kruskal(device_id = 3)
testar_significancia_kruskal(device_id = 4)
testar_significancia_kruskal(device_id = 5)
'''

## -- HEATMAPS DOS P-VALUES DO TESTE DE MANN-WHITNEY (NÃO-PARAMÉTRICO) -- ##
# Apenas para um device
# Para todos os pares de atividades
def heatmaps_modules_mannwhitney(device_id):
    cols = [13, 14, 15] # Colunas dos módulos
    module_names = ["Acelerómetro", "Giroscópio", "Magnetómetro"] # Nomes dos módulos

    # Para todas as atividades
    atividades_unicas = np.arange(1, 17)
    n_activities = len(atividades_unicas)

    # Gera um heatmap por cada tipo de módulo
    for col, mod_name in zip(cols, module_names):
        # Extrair grupos de cada atividade
        activities_groups = [
            dados[(dados[:, 11] == act) & (dados[:, 0] == device_id), col]
            for act in atividades_unicas
        ]

        p_values = np.full((n_activities, n_activities), np.nan)

        # Comparações par-a-par
        for i in range(n_activities):
            for j in range(i + 1, n_activities):
                if len(activities_groups[i]) > 0 and len(activities_groups[j]) > 0:
                    # Teste de Mann–Whitney bilateral (sem direção) -> apenas queremos saber se há uma diferença significativa
                    stat, p = mannwhitneyu(activities_groups[i], activities_groups[j], alternative = 'two-sided')
                    p_values[i, j] = p
                    p_values[j, i] = p # Simetria

        # Converter a matriz de p-values para DataFrame 
        df = pd.DataFrame(p_values, index = atividades_unicas.astype(int), columns = atividades_unicas.astype(int))

        # Plot do heatmap
        plt.figure(figsize = (12, 10))
        ax = sns.heatmap(
            df, annot = True, fmt = ".3f", linewidths = 0.5,
            cmap = "coolwarm", vmin = 0, vmax = 1,
            cbar_kws = {'label': 'p-value'}
        )

        # Adicionar contorno vermelho onde p > 0.05/(16*16) (não significativo)
        for i in range(n_activities):
            for j in range(n_activities):
                if not np.isnan(p_values[i, j]) and p_values[i, j] > 0.05 / 120:
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, fill = False, edgecolor = 'red', lw = 2))

        plt.title(f"P-values (Mann-Whitney) - {mod_name} - Device {device_id}", fontsize = 22, fontweight = 'bold', fontname = 'Trebuchet MS', color = '#c00000')
        plt.xlabel("Atividade")
        plt.ylabel("Atividade")
        # Criar legenda 
        legenda_patch = mpatches.Patch(edgecolor = 'red', facecolor = 'none', label = 'p > 0.05/120 testes', linewidth = 2)
        plt.legend(handles = [legenda_patch], loc = 'upper right', fontsize = 12, frameon = True)
        plt.show()

heatmaps_modules_mannwhitney(device_id = 2)
'''
heatmaps_modules_mannwhitney(device_id = 1)
heatmaps_modules_mannwhitney(device_id = 3)
heatmaps_modules_mannwhitney(device_id = 4)
heatmaps_modules_mannwhitney(device_id = 5)
'''

import torch
## -- DESCARREGAR O MODELO DE EXTRAÇÃO DE EMBEDDINGS -- ##
def load_model():
  ''' Loads the model from the github repo and obtains just the feature encoder. '''

  # Nome do repositório GitHub onde o modelo pré-treinado está guardado
  repo = 'OxWearables/ssl-wearables'

  # class_num não interessa para extrair features; mas o hub pede este arg
  model = torch.hub.load(repo, 'harnet5', class_num=5, pretrained=True)
  model.eval()

  # Passo crucial: ficar só com a parte auto-supervisionada
  # O README diz que há um 'feature_extractor' (pré-treinado) e um 'classifier' (não treinado). :contentReference[oaicite:14]{index=14}
  # Aqui queremos apenas o feature_extractor, que gera os embeddings
  feature_encoder = model.feature_extractor
  feature_encoder.to("cpu")
  feature_encoder.eval()

  # Devolve o feature encoder, que será usado para extrair embeddings dos dados
  return feature_encoder

## -- FAZ RESAMPLE DA FREQUÊNCIA DE AMOSTRAGEM DE 45.2 PARA 30 -- ##
def resample_to_30hz_5s(acc_xyz, fs_in_hz):
    """
    acc_xyz: matriz (N, 3) -> aceleração (x, y, z)
    fs_in_hz: frequência original (Hz)
    devolve: sinal reamostrado a 30 Hz e a nova frequência
    """
    fs_target = 30.0      # nova frequência desejada (30 Hz)
    win_size = 5          # duração do segmento em segundos

    # tempos originais (com base na frequência inicial)
    t_in = np.arange(acc_xyz.shape[0]) / fs_in_hz

    # tempos novos, uniformes, de 0 a 5s com passo 1/30
    t_out = np.arange(0, win_size, 1.0/fs_target)

    # cria matriz vazia para o sinal reamostrado (150x3)
    acc_resampled = np.zeros((len(t_out), 3), dtype=np.float32)

    # interpola cada eixo (x, y, z) para os novos tempos
    for axis in range(3):
        acc_resampled[:, axis] = np.interp(t_out, t_in, acc_xyz[:, axis])

    # devolve o sinal reamostrado e a nova frequência
    return acc_resampled, fs_target

## ------------ EXERCÍCIO 4.2 ------------ ##
## -- EXTRAI AS 110 FEATURES CONFORME O ARTIGO BODYNETS 2011 -- ##
def extract_features_110(dataset, fs, window_s, overlap, id_participant, feature_encoder):

    win_len = int(window_s * fs)
    hop = int(win_len * (1 - overlap))
    n_samples = dataset.shape[0]

    acc = dataset[:, 1:4]
    gyr = dataset[:, 4:7]
    labels = dataset[:, 11]
    
    X, y = [], []
    acc_windows = []
    feature_names = []

    # -- NOMES DAS FEATURES -- #
    base_feats = [
        "mean", "median", "std", "var", "rms", "mean_diff", "skew", "kurtosis",
        "iqr", "zero_cross_rate", "mean_cross_rate", "spec_entropy",
        "dom_freq", "spec_energy"
    ]
    axes = ["x", "y", "z"]
    sensors = ["acc", "gyr"]

    # Criar nomes automáticos das primeiras 84 features (14*3*2)
    for s in sensors:
        for ax in axes:
            for f in base_feats:
                feature_names.append(f"{s}{ax}{f}")

    # Correlações 
    corr_pairs = list(combinations(["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"], 2))
    feature_names.extend([f"corr_{a}_{b}" for (a, b) in corr_pairs])

    # Features físicas 
    feature_names.extend([
        "AI_mean", "VI_var", "SMA",
        "EVA1", "EVA2",
        "AVG", "AVH", "ARATG",
        "CAGH", "ARE", "AAE", "ID"
    ])

    # Funções auxiliares
    def rms(x): return np.sqrt(np.mean(x**2))
    def iqr(x): return np.percentile(x, 75) - np.percentile(x, 25)
    def zero_crossing_rate(x): return np.sum(np.diff(np.sign(x)) != 0) / len(x)
    def mean_crossing_rate(x): return np.sum(np.diff(np.sign(x - np.mean(x))) != 0) / len(x)
    def spectral_entropy(x, fs):
        f, Pxx = welch(x, fs = fs)
        Pxx /= np.sum(Pxx)
        return entropy(Pxx)
    def dominant_frequency(x, fs):
        f, Pxx = welch(x, fs = fs)
        return f[np.argmax(Pxx)]
    def spectral_energy(x, fs):
        f, Pxx = welch(x, fs = fs)
        return np.sum(Pxx)
    def movement_intensity(acc):
        return np.sqrt(np.sum(acc**2, axis = 1))
    def sma(data):
        return np.sum(np.abs(data)) / len(data)
    def avg_velocity(acc, fs):
        vel = np.cumsum(acc, axis = 0) / fs
        return np.mean(np.linalg.norm(vel, axis = 1))
    def eig_features(data):
        eigvals = np.linalg.eigvals(np.cov(data.T))
        return np.real(np.sort(eigvals)[::-1])[:2]  # Top 2 eigenvalues


    for start in range(0, n_samples - win_len + 1, hop):
        end = start + win_len
        lbls = labels[start:end]
        if len(np.unique(lbls)) != 1:
            continue # Descartar janelas com mais do que uma atividade

        if len(lbls) < 10:
            continue # Descartar janelas muito pequenas

        acc_w = acc[start:end]

        gyr_w = gyr[start:end]
        feats = []

        # --- + 84 Features temporais e espetrais  ---
        for sensor in [acc_w, gyr_w]:
            for i in range(3):
                x = sensor[:, i]
                feats.extend([
                    np.mean(x), np.median(x), np.std(x), np.var(x),
                    rms(x), np.mean(np.diff(x)), skew(x), kurtosis(x),
                    iqr(x), zero_crossing_rate(x), mean_crossing_rate(x),
                    spectral_entropy(x, fs), dominant_frequency(x, fs),
                    spectral_energy(x, fs)
                ])

        # --- + 15 correlações ---
        full = np.hstack((acc_w, gyr_w))
        for i, j in combinations(range(6), 2):
            feats.append(np.corrcoef(full[:, i], full[:, j])[0, 1])

        # --- + 11 features físicas ---
        mi = movement_intensity(acc_w)
        feats.append(np.mean(mi))             # AI
        feats.append(np.var(mi))              # VI
        feats.append(sma(acc_w))              # SMA
        feats.extend(eig_features(acc_w))     # EVA1, EVA2
        g_proj = acc_w[:, 2]                  # z ~ gravidade
        heading_proj = np.sqrt(acc_w[:,0]**2 + acc_w[:,1]**2)
        feats.append(np.corrcoef(g_proj, heading_proj)[0,1])  # CAGH
        feats.append(avg_velocity(acc_w, fs)) # AVG
        feats.append(np.mean(np.abs(gyr_w)))  # AVH
        feats.append(np.var(gyr_w))           # ARATG
        feats.append(np.mean(acc_w**2))       # AAE
        feats.append(np.mean(gyr_w**2))       # ARE
        feats.append(id_participant)          # Id participante

        X.append(feats)
        y.append(lbls[0])

        acc_windows.append(acc_w)

    # --- EXTRAIR EMBEDDINGS --- #
    # Reamostrar cada segmento para 30Hz (obrigatório para HARNET5)
    resampled_segments = [resample_to_30hz_5s(seg, fs)[0] for seg in acc_windows]

    # Transformar para formato [n_segmentos, canais(3), tempo(150)]
    x_all = np.transpose(np.array(resampled_segments), (0, 2, 1))
    
    # Extrair embeddings em batches
    emb_list = []
    batch_size = 5
    with torch.no_grad():
        for i in range(0, x_all.shape[0], batch_size):
            xb = torch.from_numpy(x_all[i:i+batch_size]).float()
            eb = feature_encoder(xb)         
            emb_list.append(eb.cpu().numpy())

    X_emb_p = np.concatenate(emb_list, axis=0)

    return np.array(X), np.array(y), X_emb_p, feature_names


# Frequência de amostragem
fs = 51.2
X_total, y_total = [], []
X_total_partB, y_total_partB = [], []
X_total_emb, y_total_emb = [], []
fe = load_model()

# Extração das 110 features para cada participante
for id_part in np.unique(dados[:, 12]):  # Todos os participantes
    dados_p = dados[dados[:, 12] == id_part]
    for device in np.unique(dados[:, 0]):   # Por device
        dados_p_d = dados_p[dados_p[:, 0] == device]
        X_p, y_p, X_emb, feature_names = extract_features_110(dados_p_d, fs, window_s = 5, overlap = 0.5, id_participant = id_part, feature_encoder = fe)
        print(f"\nMatriz de features + participante_id = {id_part}:", X_p.shape)
        print("features: ", len(feature_names))
        X_total.append(X_p)
        y_total.append(y_p)

        # Apenas atividades de 1 a 7 (Parte B)
        indices_atividade = [1, 2, 3, 4, 5, 6, 7]
        indices = np.where(np.isin(y_p, indices_atividade))[0]
        X_total_partB.append(X_p[indices])
        y_total_partB.append(y_p[indices])
        print(f"Matriz de features + participante_id = {id_part}:", X_p[indices].shape)

        X_total_emb.append(X_emb[indices])
        print(f"Matriz de Embeddings + participante_id = {id_part}:", X_emb[indices].shape)


X_total = np.vstack(X_total)[:, :-1]  # Excluir a última coluna (ID participante) para o PCA, relief e fisher
y_total = np.hstack(y_total)
print("Matriz de features Final:", X_total.shape)
print("Labels Finais:", y_total.shape)

# Matrizes para a Parte B (participantes 1 a 7)
X_total_partB = np.vstack(X_total_partB)
y_total_partB = np.hstack(y_total_partB)
header_str = ",".join(feature_names)
np.savetxt("Matriz_Features.csv", X_total_partB, delimiter = ",", header = header_str, comments = '')
np.savetxt("Lista_Atividades.csv", y_total_partB, delimiter = ",", header = "atividade", comments = '')

# criar um csv com timestamp, acc_x, acc_y, acc_z, timestamp, activity
emb = np.vstack(X_total_emb)
# Remover a dimensão extra para CSV
emb = np.squeeze(emb)
np.savetxt("X_Embeddings.csv", emb, delimiter = ",")

## ------------ EXERCÍCIO 4.3 ------------ ##
## -- PCA PARA REDUZIR A DIMENSIONALIDADE -- ##
def aplicar_pca(feature_set, n_components = None):

    # Normalizar as features (Z-score)
    scaler = StandardScaler()
    feature_set_norm = scaler.fit_transform(feature_set)

    # Aplicar PCA
    pca = PCA(n_components = n_components)
    pca_features = pca.fit_transform(feature_set_norm)

    # Mostrar resultados
    print("\n--- RESULTADOS DO PCA ---")
    print("Variância explicada por cada componente:")
    print(pca.explained_variance_ratio_)

    # Gráfico da variância explicada acumulada
    plt.figure(figsize = (7,4))
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_)*100, marker = 'o')

    # Linha horizontal nos 75%
    plt.axhline(y = 75, color = 'red', linestyle = '--', linewidth = 2)
    
    # Texto acima da linha
    plt.text(
        x = 37,# Posição x do texto 
        y = 75 + 2, # Um pouco acima da linha
        s = "75 %", 
        color = "red",
        fontsize = 12,
        fontweight = 'bold'
    )

    plt.xlabel('Número de Componentes')
    plt.ylabel('Variância Explicada Acumulada')
    plt.title('Análise de Componentes Principais (PCA)', fontsize = 22, fontweight = 'bold', fontname = 'Trebuchet MS', color = '#c00000')
    plt.grid(True)
    plt.xlim (0,40) # x entre [0,40]
    plt.show()

    return pca_features, pca, scaler

## ------------ EXERCÍCIO 4.4 ------------ ##
# Aplicar PCA (mantendo todas as componentes)
pca_features, pca, scaler = aplicar_pca(X_total)

# Ver quantas componentes são necessárias para explicar 75% da variância
variancia_acumulada = np.cumsum(pca.explained_variance_ratio_)
dimensoes_75 = np.argmax(variancia_acumulada >= 0.75) + 1
print(f"Número mínimo de componentes para explicar 75% da variância: {dimensoes_75}")

## ------------ EXERCÍCIO 4.4.1 ------------ ##
def exemplo_pca_instante(feature_set, scaler, pca, n_dimensoes, idx_exemplo = 0):
    """
    Obtém o vetor PCA completo de um instante específico (todas as componentes).
    Retorna:
        -> Vetor projetado no espaço PCA (todas as componentes)
    """

    # Obter o vetor original da amostra escolhida
    x_original = feature_set[idx_exemplo, :].reshape(1, -1)

    # Normalizar
    x_norm = scaler.transform(x_original)

    # Usar só as componentes principais
    W_reduced = pca.components_[:dimensoes_75, :]

    # Projetar no espaço PCA completo
    pca_features = x_norm @ W_reduced.T

    # Mostrar resultados
    print(f"\n--- Exemplo PCA Completo (instante {idx_exemplo}) ---")
    print(f"Vetor PCA (todas as componentes, dimensão {pca_features.shape[1]}):")
    print(pca_features)

    return pca_features

exemplo_pca_instante(pca_features, scaler, pca, dimensoes_75, idx_exemplo = 0)

## ------------ EXERCÍCIO 4.5 ------------ ##
def selecionar_features_fisher_reliefF(X, y, feature_names, k = 10):
    """
    Aplica Fisher Score e ReliefF para selecionar as 10 melhores features.
    """

    # Normalizar os dados
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)

    # --- Fisher Score --- #
    F_values, _ = f_classif(X_norm, y)
    idx_fisher = np.argsort(F_values)[::-1][:k]

    # --- ReliefF --- #
    fs = ReliefF(n_neighbors = 10, n_features_to_select = k, n_jobs = -1) # Paralelizado
    fs.fit(X_norm, y)
    relief_scores = fs.feature_importances_
    idx_relief = np.argsort(relief_scores)[::-1][:k]

    # ---- Mostrar resultados ----
    print("\nTop Features segundo Fisher Score:")
    for i, idx in enumerate(idx_fisher):
        print(f"  {i+1:02d}. {feature_names[idx]} — Score = {F_values[idx]:.4f}")

    print("\nTop Features segundo ReliefF:")
    for i, idx in enumerate(idx_relief):
        print(f"  {i+1:02d}. {feature_names[idx]} — Score = {relief_scores[idx]:.4f}")

    # Retorna as 3 melhores features
    top3_fish = idx_fisher[:3]
    top3_rel = idx_relief[:3]
    # Retorna as 10 melhores features
    top10_fish = idx_fisher[:10]
    top10_rel = idx_relief[:10]

    return top3_fish, top3_rel, top10_fish, top10_rel

## ------------ EXERCÍCIO 4.6 ------------ ##
top3_fish, top3_rel, top10_fish, top10_rel = selecionar_features_fisher_reliefF(X_total, y_total, feature_names, k = 10)

## ------------ EXERCÍCIO 4.6.1 ------------ ##
def obter_features_selecionadas_numInstante(feature_set, indices_melhores, instante):
    # Seleção das colunas correspondentes às 10 melhores features
    vetor_reduzido = feature_set[instante, indices_melhores]
    return vetor_reduzido

vetor_fisher = obter_features_selecionadas_numInstante(X_total, top10_fish, instante = 1)
vetor_relief = obter_features_selecionadas_numInstante(X_total, top10_rel, instante = 1)


def plot_fisher_relief_features_3D(X, y, fisher_idx, relief_idx):
    """
    Cria dois gráficos 3D com as três melhores features
    segundo Fisher Score e ReliefF, cada atividade com uma cor diferente.
    """

    fig = plt.figure(figsize = (16, 7))

    # -- Gráfico Fisher 3D -- #
    ax1 = fig.add_subplot(1, 2, 1, projection = '3d')
    sc1 = ax1.scatter(X[:, fisher_idx[0]], X[:, fisher_idx[1]], X[:, fisher_idx[2]],
                      c = y, cmap = 'tab20', s = 20, alpha = 0.8, edgecolors = 'none')
    ax1.set_title(f'Fisher Score — Features {fisher_idx}', fontname = 'Trebuchet MS', fontsize = 16, fontweight = 'bold')
    ax1.set_xlabel(f'Feature {fisher_idx[0]}')
    ax1.set_ylabel(f'Feature {fisher_idx[1]}')
    ax1.set_zlabel(f'Feature {fisher_idx[2]}')

    # -- Gráfico ReliefF 3D -- #
    ax2 = fig.add_subplot(1, 2, 2, projection = '3d')
    sc2 = ax2.scatter(X[:, relief_idx[0]], X[:, relief_idx[1]], X[:, relief_idx[2]],
                      c = y, cmap = 'tab20', s = 20, alpha = 0.8, edgecolors = 'none')
    ax2.set_title(f'ReliefF — Features {relief_idx}', fontname = 'Trebuchet MS', fontsize = 16, fontweight = 'bold')
    ax2.set_xlabel(f'Feature {relief_idx[0]}')
    ax2.set_ylabel(f'Feature {relief_idx[1]}')
    ax2.set_zlabel(f'Feature {relief_idx[2]}')

    # Barra de cores
    cbar = fig.colorbar(sc2, ax = [ax1, ax2], fraction = 0.02, pad = 0.04)
    cbar.set_label('Atividade', rotation = 270, labelpad = 15)

    plt.suptitle('Comparação das Features Selecionadas em 3D (Fisher vs ReliefF)', fontsize = 22, fontweight = 'bold', fontname = 'Trebuchet MS', color = '#c00000')
    plt.show()

plot_fisher_relief_features_3D(X_total, y_total, top3_fish, top3_rel)