Introdução:
Trabalho de classificação de 7 atividades através de seis cenários aplicados em dois splits diferentes, que foi dividido nas duas metas: "mainActivity.py" e "partB.py" sendo os ficheiros csv necessários para esta divisão que permitiu uma melhor organização do projeto.

Ficheiros:
-> mainActivity.py: Desenvolvimento de toda a primeira meta + extração de Embbedings necessária na parteB
-> Martiz_Features.csv: Contém as 110 colunas de features + 1 coluna com o ID do participante
-> Lista_Atividades: Contem uma coluna com a atividade referente a cada linha da matriz de Features
-> X_Embeddings.csv: Contém 512 colunas de embbeddings emparelhados com as mesmas linhas que as Features
-> partB.py: DesenvolvimentO da segunda meta, lê os csv anteriores para facilitar a organização do projeto
-> Resultados_KNN_31.csv: Guarda as distribuições de F1-score dos seis cenários nos dois tipos de splits 
-> partB_extras.py: Aplicados extras como o SMOTE, KNN distanced-based e Fisher Score
-> Resultados_KNN_31_SMOTE.csv : Guarda as distribuições de F1-score nos 8 cenários dos com extras

