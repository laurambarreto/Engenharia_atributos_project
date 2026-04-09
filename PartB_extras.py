## -- IMPORT DAS BIBLIOTECAS NECESSÁRIAS -- ##
import numpy as np
import matplotlib.patches as patches
import pandas as pd
from matplotlib import pyplot as plt
import math
from itertools import combinations
from scipy.stats import ttest_rel
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
import seaborn as sns
from scipy.stats import kstest
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import welch
from itertools import combinations
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from skfeature.function.similarity_based.reliefF import reliefF 
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from statsmodels.stats.anova import AnovaRM
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import BorderlineSMOTE

# --- Ler CSV com a matriz de features e vetor de atividades --- #
df_X = pd.read_csv("Matriz_Features.csv")
df_y = pd.read_csv("Lista_Atividades.csv")

vetor_participantes = df_X['ID']
print(len(vetor_participantes))
print(vetor_participantes)
df_X = df_X.drop(columns = ['ID'])
print(df_X)

# Converter para arrays NumPy
X_total = df_X.values       # matriz sem nomes das colunas
y_total = df_y.values.ravel()  # vetor 1D
print("X_total shape:", X_total.shape)
print("y_total shape:", y_total.shape)

# --- Ler CSV com a matriz de embeddings --- #
df_emb = pd.read_csv("X_Embeddings.csv", header = None)
# Converter para arrays NumPy
X_emb = df_emb.values       # matriz sem nomes das colunas
print("X_Emb shape:", X_emb.shape)

## ------------ EXERCÍCIO 1.1 ------------ ##
# Verificar balanceamento das atividades
def desbalanceamento():
    # Obter atividades únicas e suas contagens
    atividades_unicas, contagens = np.unique(y_total, return_counts = True)

    # Plot gráfico de barras
    plt.figure(figsize = (6, 5))
    plt.bar(atividades_unicas, contagens, color = 'skyblue', edgecolor = 'black', zorder = 3)
    
    # Adicionar contagens acima das barras
    for x, y in zip(atividades_unicas, contagens):
        plt.text(x, y + max(contagens)*0.005, str(y), ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.grid(True, which = 'major', linestyle = '--', linewidth = 0.7, alpha = 0.4, zorder = 0)
    plt.title("Distribuição das Atividades", fontname = 'Trebuchet MS', fontsize = 16, fontweight = 'bold') 
    plt.xlabel("Atividade")
    plt.ylabel("Contagem")  
    plt.show()

desbalanceamento()


## ------------ EXERCÍCIO 1.2 ------------ ## 
# Função para criar k novas amostras sintéticas numa atividade 
def SMOTE(X, y, n_novos, k_vizinhos, atividade): 
    X_atividade = X[(y == atividade)]
    
    n_amostras = X_atividade.shape[0] 
    X_sinteticas = [] 
    # Amostras para gerar por ponto da atividade
    N_T = math.ceil(n_novos / n_amostras)

    # Para cada amostra na atividade vamos gerar N/T pontos sintéticos
    for i in range(n_amostras): 
        amostra_base = X_atividade[i] 
        geradas = 0 
        while geradas < N_T and len(X_sinteticas) < n_novos:
            # Calcular distâncias aos outros pontos nesta atividade
            distancias = np.linalg.norm(X_atividade - amostra_base, axis = 1)
            indices_ordenados = np.argsort(distancias)
            vizinhos = indices_ordenados[1:k_vizinhos+1] # Ignorar a distância ao próprio ponto
           
            # Escolher um aleatorio entre os k vizinhos mais próximos
            vizinho_aleatorio = np.random.choice(vizinhos)
            
            # Obter o vetor entre a amostra base e o vizinho aleatório 
            vetor = X_atividade[vizinho_aleatorio] - amostra_base 
            
            # Gerar nova amostra sintética 
            fator = np.random.rand() 
            nova_amostra = amostra_base + fator * vetor 
            
            X_sinteticas.append(nova_amostra)
            geradas += 1

    return np.array(X_sinteticas)


# ------------ EXERCÍCIO 1.3 ------------ #
# Plot 2D das amostras originais e sintéticas (com as duas primeiras features no eixo X e Y)
def plot_amostras_2D(X_total, y_total, X, y, atividade, n_sinteticas, k_vizinhos):
    # Gerar amostras sintéticas para a atividade
    X_sinteticas = SMOTE(X, y, n_novos = n_sinteticas, k_vizinhos = k_vizinhos, atividade = atividade)
    
    # Obter as atividades únicas
    atividades_unicas = np.unique(y)
    
    # Paleta de 7 cores para as atividades
    cores = [
        "#FF6F61",  # 1 - Coral
        "#FF69C1",  # 2 - Roxo
        "#EF75FF",  # 3 - Verde médio
        "#B2FF66",  # 4 - Verde claro (atividade a destacar)
        "#FFA500",  # 5 - Laranja
        "#ECF665",  # 6 - Verde água
        "#8067FD"   # 7 - Rosa claro
    ]
    # -- Figura que mostra só os dados do participante 3 -- #
    plt.figure(figsize = (10, 6))
    # Plot de cada atividade com cor diferente
    for i, act in enumerate(atividades_unicas):
        X_act = X[y ==  act]

        if act != atividade:
            plt.scatter(X_act[:, 0], X_act[:, 60], color = cores[i], label = f'Atividade {int(act)}', alpha = 0.5, zorder = 2)
        else:
            # Plot das amostras da atividade na camada da frente   
            plt.scatter(X_act[:, 0], X_act[:, 60], color = cores[i], label = f'Atividade {int(atividade)}', alpha = 0.5, zorder = 3)

            # Plot das amostras sintéticas (verde escuro, mais visíveis)
            plt.scatter(X_sinteticas[:, 0], X_sinteticas[:, 60],
            color = "darkgreen", marker = 'X', s = 75,
            label = f'Atividade {atividade} ({n_sinteticas} Amostras Sintéticas)', zorder = 3) 

    # Estilo e legendas
    plt.title(f'Amostras Originais e Sintéticas - Atividade {atividade}',
              fontname = 'Trebuchet MS', fontsize = 16, fontweight = 'bold')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 61')
    plt.legend()
    plt.grid(True, which = 'major', linestyle = '--', linewidth = 0.7, alpha = 0.4, zorder = 0)
    plt.show()

    # -- Figura que mostra os dados de todos os participantes -- #
    plt.figure(figsize = (10, 6))
    # Plot de cada atividade com cor diferente
    for i, act in enumerate(atividades_unicas):
        X_act = X_total[y_total ==  act]

        if act != atividade:
            plt.scatter(X_act[:, 0], X_act[:, 60], color = cores[i], label = f'Atividade {int(act)}', alpha = 0.5, zorder = 2)
        else:
            # Plot das amostras da atividade na camada da frente   
            plt.scatter(X_act[:, 0], X_act[:, 60], color = cores[i], label = f'Atividade {int(atividade)}', alpha = 0.5, zorder = 3)

            # Plot das amostras sintéticas (verde escuro, mais visíveis)
            plt.scatter(X_sinteticas[:, 0], X_sinteticas[:, 60],
            color = "darkgreen", marker = 'X', s = 75,
            label = f'Atividade {atividade} ({n_sinteticas} Amostras Sintéticas)', zorder = 3) 

    # Estilo e legendas
    plt.title(f'Amostras Originais e Sintéticas - Atividade {atividade}',
              fontname = 'Trebuchet MS', fontsize = 16, fontweight = 'bold')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 61')
    plt.legend()
    plt.grid(True, which = 'major', linestyle = '--', linewidth = 0.7, alpha = 0.4, zorder = 0)
    plt.show()

X_part3 = X_total[vetor_participantes == 3]
y_part3 = y_total[vetor_participantes == 3]
plot_amostras_2D(X_total, y_total, X_part3, y_part3, atividade = 4, n_sinteticas = 3, k_vizinhos = 5)


## ---- OBTER A MATRIZ DO PCA APLICADO ÀS FEATURES ---- ##
def aplicar_pca(feature_set):

    # Normalizar as features (Z-score)
    scaler = StandardScaler()
    feature_set_norm = scaler.fit_transform(feature_set)

    # Aplicar PCA
    pca = PCA()
    pca_features = pca.fit_transform(feature_set_norm)

    # Ver quantas componentes são necessárias para explicar 75% da variância
    variancia_acumulada = np.cumsum(pca.explained_variance_ratio_)
    dimensoes_90 = np.argmax(variancia_acumulada >= 0.90) + 1
    print(f"Número mínimo de componentes para explicar 90% da variância: {dimensoes_90}")

    return pca_features, pca, scaler, dimensoes_90


## ---- OBTER A MATRIZ DO PCA no train, validation, train + validation e teste ---- ##
def aplicar_pca_train_val_test(X_train, X_val, X_train_val, X_test, var_exp = 0.90):
    """
    Aplica PCA apenas no treino, transforma validação, treino+validação e teste,
    retornando também scaler, pca e n_comp.
    """

    # Normalizar pelo treino
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_val_norm = scaler.transform(X_val)
    X_train_val_norm = scaler.transform(X_train_val)
    X_test_norm = scaler.transform(X_test)

    # PCA ajustado só no treino
    pca = PCA()
    X_train_pca_full = pca.fit_transform(X_train_norm)

    # Determinar nº componentes >= 90% variância
    var_acum = np.cumsum(pca.explained_variance_ratio_)
    n_comp = np.argmax(var_acum >= var_exp) + 1

    # Gerar projeções reduzidas
    X_train_pca = X_train_pca_full[:, :n_comp]
    X_val_pca = X_val_norm @ pca.components_[:n_comp, :].T
    X_train_val_pca = X_train_val_norm @ pca.components_[:n_comp, :].T
    X_test_pca = X_test_norm @ pca.components_[:n_comp, :].T

    # Retornar tudo o que precisas
    return X_train_pca, X_val_pca, X_train_val_pca, X_test_pca


## ---- OBTER A MATRIZ DO FISHER no train, validation, train + validation e teste ---- ##
def fisher_score(X_train, y_train, X_val, X_train_val, X_test, top_k = 15):
    # Normalizar apenas com treino
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_val_norm = scaler.transform(X_val)
    X_train_val_norm = scaler.transform(X_train_val)
    X_test_norm = scaler.transform(X_test)
    
    # Seleção de features automática
    selector = SelectKBest(score_func=f_classif, k = top_k)
    X_train_reduced = selector.fit_transform(X_train_norm, y_train)
    
    # Aplicar seleção aos outros conjuntos
    X_val_reduced = selector.transform(X_val_norm)
    X_train_val_reduced = selector.transform(X_train_val_norm)
    X_test_reduced = selector.transform(X_test_norm)
    
    return X_train_reduced, X_val_reduced, X_train_val_reduced, X_test_reduced


## ---- OBTER A MATRIZ DO RELIEF  no train, validation, train + validation e teste ---- ##
def relief_manual(X_train, y_train, X_val, X_train_val, X_test, top_k=15):
    """
    Seleciona as top_k features com o método ReliefF,
    usando apenas o conjunto de treino.
    Retorna:
        - X_train_reduced
        - X_val_reduced
        - X_train_val_reduced
        - X_test_reduced
        - idx_top (índices das melhores features)
    """

    # Número de amostras e features do treino
    n_samples, n_features = X_train.shape
    
    # Normalizar os dados
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_val_norm = scaler.transform(X_val)
    X_train_val_norm = scaler.transform(X_train_val)
    X_test_norm = scaler.transform(X_test)

    # Calcular matriz de distâncias entre todas as amostras do treino 
    D = cdist(X_train_norm, X_train_norm, metric = "euclidean")
    np.fill_diagonal(D, np.inf)  # ignora distância do ponto a si próprio

    # Guardar, para cada classe, os índices das suas amostras 
    class_indices = {c: np.where(y_train == c)[0] for c in np.unique(y_train)}

    # Arrays onde vamos guardar o nearest hit e o nearest miss de cada amostra 
    nearHit = np.zeros(n_samples, dtype=int)
    nearMiss = np.zeros(n_samples, dtype=int)

    # Para cada classe, encontrar nearest hit e nearest miss de todos os pontos
    for c, idx_same in class_indices.items():

        # Amostras de outras classes
        idx_diff = np.setdiff1d(np.arange(n_samples), idx_same, assume_unique = True)

        # Submatriz distâncias dentro da mesma classe
        subD_same = D[np.ix_(idx_same, idx_same)]
        # Submatriz distâncias para classes diferentes
        subD_diff = D[np.ix_(idx_same, idx_diff)]

        # Ignorar diagonal
        np.fill_diagonal(subD_same, np.inf)

        # Nearest hit: ponto mais próximo da mesma classe
        nearHit[idx_same] = idx_same[np.argmin(subD_same, axis = 1)]

        # Nearest miss: ponto mais próximo de classes diferentes
        nearMiss[idx_same] = idx_diff[np.argmin(subD_diff, axis = 1)]

    # --- Cálculo vetorizado do peso de cada feature ---
    # (média das diferenças quadráticas hit vs miss)
    weights = np.mean(
        (X_train_norm - X_train_norm[nearMiss])**2 - (X_train_norm - X_train_norm[nearHit])**2,
        axis = 0
    )

    # Selecionar as top_k features com maior peso 
    idx_top = np.argsort(weights)[-top_k:][::-1]

    # Reduzir treino, validação e treino+validação para estas features 
    X_train = X_train_norm[:, idx_top]
    X_val = X_val_norm[:, idx_top]
    X_train_val = X_train_val_norm[:, idx_top]
    X_test = X_test_norm[:, idx_top]

    return X_train, X_val, X_train_val, X_test


## ---- OBTER COLUNAS COM MAIOR SCORE NO RELIEF APLICADO ÀS FEATURES ---- ##
def selecionar_features_reliefF(X, y, k_vizinhos, k = 15):
    """
    Aplica ReliefF para selecionar as 15 melhores features.
    """

    # Normalizar os dados
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)

    # --- ReliefF --- #
    relief_scores = reliefF(X_norm, y, n_neighbors = k_vizinhos)
    idx_relief = np.argsort(relief_scores)[::-1][:k]
    matriz_relief = X[:, idx_relief]
    print(idx_relief)
    matriz_relief = X[:,idx_relief]
    
    return matriz_relief


## -- 1º TVT - WITHIN SUBJECT (cada participante aparece em todos os conjuntos) -- ##
def split_within_subject(X, y, X_emb, vetor_participantes):
    train_X, val_X, test_X = [], [], []
    emb_train_X, emb_val_X, emb_test_X = [], [], []
    train_y, val_y, test_y = [], [], []
    

    participantes_unicos = np.unique(vetor_participantes)

    for subj in participantes_unicos:
        # Índices deste participante
        idx = vetor_participantes == subj
        X_subj = X[idx]
        y_subj = y[idx]
        X_emb_subj = X_emb[idx]

        # Número de amostras
        n = len(X_subj)
        perm = np.random.permutation(n)

        # 60-20-20
        train_end = int(0.6 * n)
        val_end = int(0.8 * n)

        # Divisão 60-20-20 nas features
        train_X.append(X_subj[perm[:train_end]])
        val_X.append(X_subj[perm[train_end:val_end]])
        test_X.append(X_subj[perm[val_end:]])

        train_y.append(y_subj[perm[:train_end]])
        val_y.append(y_subj[perm[train_end:val_end]])
        test_y.append(y_subj[perm[val_end:]])
        
        # Divisão 60-20-20 nos embeddings
        emb_train_X.append(X_emb_subj[perm[:train_end]])
        emb_val_X.append(X_emb_subj[perm[train_end:val_end]])
        emb_test_X.append(X_emb_subj[perm[val_end:]])


    # Concatenar arrays de todos os participantes
    X_train = np.vstack(train_X)
    X_val = np.vstack(val_X)
    X_test = np.vstack(test_X)

    X_train_emb = np.vstack(emb_train_X)
    X_val_emb = np.vstack(emb_val_X)
    X_test_emb = np.vstack(emb_test_X)

    y_train = np.hstack(train_y)
    y_val = np.hstack(val_y)
    y_test = np.hstack(test_y)

    return X_train, X_val, X_test, y_train, y_val, y_test, X_train_emb, X_val_emb, X_test_emb

## -- 2º TVT - BETWEEN SUBJECTS (cada participante só aparece num conjunto) -- ##
def split_between_subjects(X, y, X_emb, vetor_participantes):
    # Baralhar aleatoriamente a lista de participantes
    subjects = list(vetor_participantes.unique())
    np.random.shuffle(subjects)

    # Selecionar 9 participantes para treino, 3 para validação e 3 para teste
    train_subj = subjects[:9]
    val_subj   = subjects[9:12]
    test_subj  = subjects[12:15]

    # Criar os conjuntos garantindo que não há participantes repetidos
    train_df = X[vetor_participantes.isin(train_subj)]
    val_df = X[vetor_participantes.isin(val_subj)]
    test_df = X[vetor_participantes.isin(test_subj)]
    y_train = y[vetor_participantes.isin(train_subj)]
    y_val = y[vetor_participantes.isin(val_subj)]
    y_test = y[vetor_participantes.isin(test_subj)]

    X_train_emb = X_emb[vetor_participantes.isin(train_subj)]
    X_val_emb   = X_emb[vetor_participantes.isin(val_subj)]
    X_test_emb  = X_emb[vetor_participantes.isin(test_subj)]

    return train_df, val_df, test_df, y_train, y_val, y_test, X_train_emb, X_val_emb, X_test_emb


# ------------ EXERCÍCIO 4.1 ------------ #
def KNN_modelo_implementado (X_train, y_train, X_test, k):
    y_pred = [] 

    # percorre todas as amostras do conjunto de teste
    for i in range(len(X_test)):
        # calcula a distância entre a amostra de teste e todas as amostras de treino
        distances = np.linalg.norm(X_train - X_test[i], axis=1)
        
        # obtém os índices das k amostras de treino mais próximas 
        k_indices = np.argsort(distances)[:k]
        
        # recolhe as etiquetas (classes) dessas k amostras mais próximas
        k_nearest_labels = [y_train[j] for j in k_indices]
        
        # escolhe a classe mais comum entre os vizinhos (a que aparece mais vezes)
        most_common = max(set(k_nearest_labels), key = k_nearest_labels.count)
        
        # adiciona essa classe como previsão para a amostra atual
        y_pred.append(most_common)

    return y_pred


# ------------ EXERCÍCIO 4.2 ------------ #
def KNN_modelo_output (true_labels, predict_labels):

    print("-------- MODELO KNN NO TESTE --------")
    print("Accuracy:", accuracy_score(true_labels, predict_labels))
    print("Relatório de classificação:")
    print(classification_report(true_labels, predict_labels, labels = [1, 2, 3, 4, 5 ,6, 7], target_names = ["1", "2", "3", "4", "5", "6", "7"]))
    print()
    
    # MATRIZ DE CONFUSÃO DO MODELO KNN 
    cm = confusion_matrix(true_labels, predict_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [1, 2, 3, 4, 5 ,6, 7])
    disp.plot(cmap = plt.cm.Blues, values_format = 'd', text_kw = {'fontsize':16})  
    plt.title("Matriz de Confusão", fontsize = 22)
    plt.xlabel("Classe Prevista", fontsize = 14)
    plt.ylabel("Classe Verdadeira", fontsize = 14)
    plt.show()
    

## -- MODELO KNN USADO SÓ PARA O K MELHOR NO TESTE -- ##
def KNN_modelo (X_train, y_train, X_test, y_test, k):
    KNN = KNeighborsClassifier (k)
    KNN.fit (X_train, y_train)
    
    y_pred = KNN.predict (X_test)
    f1 = f1_score(y_test, y_pred, average = 'macro')

    KNN_modelo_output(y_test, y_pred)
    return f1


## -- MODELO KNN USADO NA VALIDAÇÃO DOS K -- ##
def KNN_modelo_val (X_train, y_train, X_val, y_val):
    f1_scores = []
    # Números de k a serem testados no treino validação
    for k in range(1, 15, 3):
        KNN = KNeighborsClassifier (k, weights='distance', metric='euclidean')
        KNN.fit (X_train, y_train)
        
        y_pred = KNN.predict (X_val)
        f1 = f1_score(y_val, y_pred, average = 'macro')
        f1_scores.append(f1)
        print(f"K = {k}: {f1}")

    # índice do maior f1
    idx = np.argmax(f1_scores)

    # converter o índice para k: 1, 4, 7, 10, 13
    melhor_k = 1 + 3 * idx
    print(f"Melhor k = {melhor_k}")
    return melhor_k


## -- NORMALIZAR OS CONJUNTOS ANTES DO KNN -- ##
def normalizar_train_val_test(X_train, X_val, X_test, X_train_val):
    """
    Recebe X_train, X_val, X_test e devolve todos normalizados usando StandardScaler.
    O scaler é ajustado apenas no treino e aplicado ao val e teste.
    """
    # Normalizar treino + validação
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Nomrmalizar (treino + validação) + teste
    scaler1 = StandardScaler()
    X_train_val_scaled = scaler1.fit_transform(X_train_val)
    X_test_scaled = scaler1.transform(X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled, X_train_val_scaled


## -- TREINAR E VALIDAR COM VÁRIOS K NOS 12 CENÁRIOS E TESTAR COM MELHOR K -- ##
def k_tunning_in_train_val(X, y, X_emb, vetor_part):
    # Criar dataframe vazio com 20 linhas (uma por trial)
    df_resultados = pd.DataFrame({
        'trial': range(31),
        'features_BS': [None]*31,
        'features_pca_BS': [None]*31,
        'features_relief_BS': [None]*31,
        'features_fisher_BS': [None]*31,
        'embeddings_BS': [None]*31,
        'embeddings_pca_BS': [None]*31,
        'embeddings_relief_BS': [None]*31,
        'embeddings_fisher_BS': [None]*31
    })

    for trial in range(1, 32):
        print(f"\nTRIAL {trial}:")
        # -------- SPLIT --------
        X_train1, X_val1, X_test1, y_train1, y_val1, y_test1, X_train_emb1, X_val_emb1, X_test_emb1 = split_between_subjects(X, y, X_emb, vetor_part)

        # -------- CONCATENAR TRAIN+VAL --------
        X_train_val1 = np.vstack([X_train1, X_val1])
        y_train_val1 = np.hstack([y_train1, y_val1])

        X_train_val_emb1 = np.vstack([X_train_emb1, X_val_emb1])

        # --- GUARDAR LABELS ORIGINAIS PARA SMOTE NOS EMBEDDINGS ---
        y_train_emb1 = y_train1.copy()
        y_train_val_emb1 = y_train_val1.copy()

        # -------- SMOTE FEATURES --------
        X_train1, y_train1 = BorderlineSMOTE().fit_resample(X_train1, y_train1)
        X_train_val1, y_train_val1 = BorderlineSMOTE().fit_resample(X_train_val1, y_train_val1)

        # -------- SMOTE EMBEDDINGS --------
        X_train_emb1, y_train_emb1 = BorderlineSMOTE().fit_resample(X_train_emb1, y_train_emb1)
        X_train_val_emb1, y_train_val_emb1 = BorderlineSMOTE().fit_resample(X_train_val_emb1, y_train_val_emb1)

        ## --- PCA FEATURES E EMBEDDINGS --- ##
        # TVT 9-3-3 -> Nas Features e Embeddings
        X_train_pca_feat1, X_val_pca_feat1, X_train_val_pca_feat1, X_test_pca_feat1 = aplicar_pca_train_val_test(X_train1, X_val1, X_train_val1, X_test1)
        X_train_pca_emb1, X_val_pca_emb1, X_train_val_pca_emb1, X_test_pca_emb1 = aplicar_pca_train_val_test(X_train_emb1, X_val_emb1, X_train_val_emb1, X_test_emb1)

        ## --- RELIEF-F FEATURES EMBEDDINGS --- ##
        # TVT 9-3-3 -> Nas Features e Embeddings
        X_train_relief_feat1, X_val_relief_feat1, X_train_val_relief_feat1, X_test_relief_feat1 = relief_manual(X_train1, y_train1, X_val1, X_train_val1, X_test1, top_k = 15) 
        X_train_relief_emb1, X_val_relief_emb1, X_train_val_relief_emb1, X_test_relief_emb1 = relief_manual(X_train_emb1, y_train1, X_val_emb1, X_train_val_emb1, X_test_emb1, top_k = 15) 
        
        ## --- Fisher Score FEATURES EMBEDDINGS --- ##
        # TVT 9-3-3 -> Nas Features e Embeddings
        X_train_fisher_feat1, X_val_fisher_feat1, X_train_val_fisher_feat1, X_test_fisher_feat1  = fisher_score(X_train1, y_train1, X_val1, X_train_val1, X_test1, top_k = 15) 
        X_train_fisher_emb1, X_val_fisher_emb1, X_train_val_fisher_emb1, X_test_fisher_emb1 = fisher_score(X_train_emb1, y_train1, X_val_emb1, X_train_val_emb1, X_test_emb1, top_k = 15) 

        ## ----- NORMALIZAR ANTES DO KNN ----- ##
        # --- PCA FEATURES E EMBEDDINGS ---
        # TVT 9-3-3
        X_train_pca_feat1, X_val_pca_feat1, X_test_pca_feat1, X_train_val_pca_feat1 = normalizar_train_val_test(X_train_pca_feat1, X_val_pca_feat1, X_test_pca_feat1, X_train_val_pca_feat1)
        X_train_pca_emb1, X_val_pca_emb1, X_test_pca_emb1, X_train_val_pca_emb1 = normalizar_train_val_test(X_train_pca_emb1, X_val_pca_emb1, X_test_pca_emb1, X_train_val_pca_emb1)
        
        # --- RELIEF-F FEATURES E EMBEDDINGS ---
        # TVT 9-3-3
        X_train_relief_feat1, X_val_relief_feat1, X_test_relief_feat1, X_train_val_relief_feat1 = normalizar_train_val_test(X_train_relief_feat1, X_val_relief_feat1, X_test_relief_feat1, X_train_val_relief_feat1)
        X_train_relief_emb1, X_val_relief_emb1, X_test_relief_emb1, X_train_val_relief_emb1 = normalizar_train_val_test(X_train_relief_emb1, X_val_relief_emb1, X_test_relief_emb1, X_train_val_relief_emb1)

        # --- Fisher Score FEATURES E EMBEDDINGS ---
        # TVT 9-3-3
        X_train_fisher_feat1, X_val_fisher_feat1, X_test_fisher_feat1, X_train_val_fisher_feat1 = normalizar_train_val_test(X_train_fisher_feat1, X_val_fisher_feat1, X_test_fisher_feat1, X_train_val_fisher_feat1)
        X_train_fisher_emb1, X_val_fisher_emb1, X_test_fisher_emb1, X_train_val_fisher_emb1 = normalizar_train_val_test(X_train_fisher_emb1, X_val_fisher_emb1, X_test_fisher_emb1, X_train_val_fisher_emb1)

        def auxiliar(nome, trial, X_train, y_train, X_val, y_val, X_train_val, y_train_val, X_test, y_test):
            print(f"\n----- Trial = {trial}, {nome} -----")
            best_k = KNN_modelo_val (X_train, y_train, X_val, y_val)
            f1_test_features = KNN_modelo(X_train_val, y_train_val, X_test, y_test, best_k)
            df_resultados.loc[trial, nome] = f1_test_features

        ## --- TVT 9-3-3 -> Obter a F1-score no teste com o melhor k da validação nos seis cenários --- ##
        # Features
        auxiliar("features_BS", trial, X_train1, y_train1, X_val1, y_val1, X_train_val1, y_train_val1, X_test1, y_test1)
        auxiliar("features_pca_BS", trial, X_train_pca_feat1, y_train1, X_val_pca_feat1, y_val1, X_train_val_pca_feat1, y_train_val1, X_test_pca_feat1, y_test1)
        auxiliar("features_relief_BS", trial, X_train_relief_feat1, y_train1, X_val_relief_feat1, y_val1, X_train_val_relief_feat1, y_train_val1, X_test_relief_feat1, y_test1)
        auxiliar("features_fisher_BS", trial, X_train_fisher_feat1, y_train1, X_val_fisher_feat1, y_val1, X_train_val_fisher_feat1, y_train_val1, X_test_fisher_feat1, y_test1)
        # Embeddings
        auxiliar("embeddings_BS", trial, X_train_emb1, y_train_emb1, X_val_emb1, y_val1, X_train_val_emb1, y_train_val_emb1, X_test_emb1, y_test1)
        auxiliar("embeddings_pca_BS", trial, X_train_pca_emb1, y_train_emb1, X_val_pca_emb1, y_val1, X_train_val_pca_emb1, y_train_val_emb1, X_test_pca_emb1, y_test1)
        auxiliar("embeddings_relief_BS", trial, X_train_relief_emb1, y_train_emb1, X_val_relief_emb1, y_val1, X_train_val_relief_emb1, y_train_val_emb1, X_test_relief_emb1, y_test1)
        auxiliar("embeddings_fisher_BS", trial, X_train_fisher_emb1, y_train_emb1, X_val_fisher_emb1, y_val1, X_train_val_fisher_emb1, y_train_val_emb1, X_test_fisher_emb1, y_test1)
        
    # salvar
    df_resultados.to_csv("Resultados_KNN_31_SMOTE.csv", index = False) 

#k_tunning_in_train_val(X_total, y_total, X_emb, vetor_participantes)


## -- LER UM ÚNICO CSV COM TODAS AS DISTRIBUIÇÕES DE F1 -- ##
df = pd.read_csv("Resultados_KNN_31_SMOTE.csv")

# ----- BETWEEN-SUBJECTS -----
f1_features_BS = df["features_BS"].values
f1_features_pca_BS = df["features_pca_BS"].values
f1_features_relief_BS = df["features_relief_BS"].values
f1_features_fisher_BS = df["features_fisher_BS"].values
f1_embeddings_BS = df["embeddings_BS"].dropna().values
f1_embeddings_pca_BS = df["embeddings_pca_BS"].values
f1_embeddings_relief_BS = df["embeddings_relief_BS"].values
f1_embeddings_fisher_BS = df["embeddings_fisher_BS"].values

# Criar a lista de listas para testes estatisticos
listas_f1_BS = [
    f1_features_BS,
    f1_features_pca_BS,
    f1_features_relief_BS,
    f1_features_fisher_BS,
]
listas_f1_BS_emb = [
    f1_embeddings_BS,
    f1_embeddings_pca_BS,
    f1_embeddings_relief_BS,
    f1_embeddings_fisher_BS
]

nomes = ["features_BS", "features_pca_BS", "features_relief_BS", "features_fisher_BS"]
nomes_emb = ["emb_BS", "emb_pca_BS", "emb_relief_BS", "emb_fisher_BS"]
nomes_total = nomes + nomes_emb

## -- TESTE KOLMOGOROV PARA VER SE HÁ PELO MENOS UMA DIFERENÇA SIGNIFICATIVA NOS 12 CENÁRIOS -- ##
def heatmap_KS(listas_feats, listas_emb, nomes):
    # Junta todos os cenários (features + embeddings)
    listas = listas_feats + listas_emb
    n = len(listas)

    # Matriz com 1 linha e n colunas
    matriz = np.zeros((1, n))

    # Cálculo dos p-values KS
    for j, dados in enumerate(listas):
        mu, sigma = np.mean(dados), np.std(dados, ddof=1)
        _, p = kstest(dados, 'norm', args=(mu, sigma))
        matriz[0, j] = p

    plt.figure(figsize=(12, 2.5))
    img = plt.imshow(matriz, cmap="coolwarm", aspect="auto")

    plt.xticks(range(n), nomes, rotation=45, ha="right")
    plt.yticks([0], ["p-values KS"])

    # Valores dentro das células + retângulos
    for j in range(n):
        plt.text(j, 0, f"{matriz[0,j]:.3f}", ha="center", va="center")
        rect = patches.Rectangle((j - 0.5, -0.5), 1, 1,
                                 fill=False, edgecolor='black', lw=1)
        plt.gca().add_patch(rect)

    plt.colorbar(img, label="p-value KS")

    plt.title("p-values do Teste KS para os 8 cenários (Split Between Subjects)")
    plt.tight_layout()
    plt.show()

heatmap_KS(listas_f1_BS, listas_f1_BS_emb, nomes_total)


## -- TESTE FRIEDMAN PARA VER SE HÁ PELO MENOS UMA DIFEREÇA SIGNIFICATIVA NOS 12 CENÁRIOS -- ##
def teste_anova_emparelhada(lista_de_listas, nomes_cenarios=None):
    """
    ANOVA de medidas repetidas (dentro dos mesmos sujeitos) para comparar
    12 cenários com dados emparelhados.
    """
    n_cenarios = len(lista_de_listas)
    n_sujeitos = len(lista_de_listas[0])

    n_cenarios = len(lista_de_listas)
    n_sujeitos = len(lista_de_listas[0])


    # Construir DataFrame largo
    df = pd.DataFrame({nomes_cenarios[i]: lista_de_listas[i] for i in range(n_cenarios)})
    df["Subject"] = range(1, n_sujeitos + 1)

    # Converter para formato longo (exigido pela AnovaRM)
    df_long = df.melt(id_vars="Subject", var_name="Cenario", value_name="Valor")

    # ANOVA de medidas repetidas
    anova = AnovaRM(data=df_long, depvar="Valor", subject="Subject",
                    within=["Cenario"]).fit()

    p = anova.anova_table["Pr > F"].iloc[0]

    if p < 0.05:
        print(f"p-valor = {p:.4f} → diferenças significativas em pelo menos um cenário.")
    else:
        print(f"p-valor = {p:.4f} → não há diferenças estatisticamente significativas.")

print("\n--- ANOVA EMPARELHADA NAS FEATURES DO SPLIT BS: ---")
teste_anova_emparelhada(listas_f1_BS, nomes)

print("\n--- ANOVA EMPARELHADA NOS EMBEDDINGS DO SPLIT BS: ---")
teste_anova_emparelhada(listas_f1_BS_emb, nomes_emb)


## -- TESTE WILCOXON PARA VER QUAIS PARES DE CENÁRIOS TÊM DIFEREÇAS SIGNIFICATIVAS -- ##
def heatmap_ttest_emparelhado(lista_de_listas, nomes_cenarios, alpha, mod_name="Cenários"):
    """
    Aplica t-test pareado entre todos os pares de cenários e gera heatmap dos p-valores.

    lista_de_listas: lista de listas/arrays com medidas (ex: F1) de cada cenário
    nomes_cenarios: lista com nomes dos cenários
    alpha: nível de significância (ex: Bonferroni já aplicado)
    mod_name: título do heatmap
    """
    n = len(lista_de_listas)
    p_values = np.full((n, n), np.nan)

    # Teste t pareado para todos os pares
    for i, j in combinations(range(n), 2):
        if len(lista_de_listas[i]) > 0 and len(lista_de_listas[j]) > 0:
            stat, p = ttest_rel(lista_de_listas[i], lista_de_listas[j])
            p_values[i, j] = p
            p_values[j, i] = p  # simetria

    # DataFrame para seaborn
    df = pd.DataFrame(p_values, index=nomes_cenarios, columns=nomes_cenarios)

    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(df, annot=True, fmt=".3f", linewidths=0.5,
                     cmap="coolwarm", vmin=0, vmax=1,
                     cbar_kws={'label': 'p-value'})

    # Contorno a vermelho quando *não* é significativo
    for i in range(n):
        for j in range(n):
            if not np.isnan(p_values[i, j]) and p_values[i, j] > alpha:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False,
                                           edgecolor='red', lw=1.5))

    # Legenda
    patch = mpatches.Patch(edgecolor='red', facecolor='none', lw=2,
                           label=f'p-value > 0.05/6')
    plt.legend(handles=[patch], loc='upper right', title="Sem diferenças significativas:")

    plt.title(f"Paired t-test p-values - {mod_name}")
    plt.yticks(rotation=0, ha="right")
    plt.show()

heatmap_ttest_emparelhado(listas_f1_BS, nomes_cenarios = nomes, alpha = 0.05 / 6, mod_name = "TVT between participants")  # Bonferroni


## -- BOXPLOTS DE DISTRIBUIÇÕES DOS F1 DOS 12 CENÁRIOS -- ##
def boxplot_cenarios(lista_de_listas, nomes_cenarios=None, mod_name="Cenários"):
    """
    Gera um boxplot comparativo dos valores de F1 de cada cenário.

    lista_de_listas: lista de arrays/listas com F1 de cada cenário
    nomes_cenarios: lista de nomes para os cenários
    mod_name: título do gráfico
    """

    # Transformar em DataFrame para seaborn
    dados = []
    cenarios = []
    for i, valores in enumerate(lista_de_listas):
        dados.extend(valores)
        cenarios.extend([nomes_cenarios[i]] * len(valores))

    df = pd.DataFrame({"Cenário": cenarios, "F1": dados})

    plt.figure(figsize = (12,6))
    sns.boxplot(x = "Cenário", y = "F1", data = df, zorder = 2)
    plt.grid(zorder = 1)
    plt.axhline(1/7, color = 'red', linestyle='--', linewidth = 1.5, zorder = 1)
    # Adicionar texto acima da linha
    plt.text(
        x = 0.5,               # posição horizontal (0 = esquerda, 1 = direita, 0.5 = centro)
        y = 1/7 + 0.01,        # ligeiramente acima da linha
        s = "1/7 atividades",  # texto
        ha = 'center',         # alinhamento horizontal
        va = 'bottom',         # alinhamento vertical
        fontsize = 11,
        color = 'red'
    )
    plt.title(f"Distribuição de F1 por cenário - {mod_name}")
    plt.xticks(rotation = 20)
    plt.ylabel("F1 Score")
    plt.ylim(0.05, 0.85)
    plt.show()

boxplot_cenarios(listas_f1_BS, nomes_cenarios = nomes)
