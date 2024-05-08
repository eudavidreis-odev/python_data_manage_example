# Projeto de Estudos: Python Aplicado a Big Data

Este projeto faz parte dos meus estudos na faculdade, onde estou explorando Python aplicado a Big Data. O objetivo é utilizar técnicas de aprendizado de máquina para análise de dados.

## Descrição

Neste projeto, utilizamos o conjunto de dados Iris e técnicas de aprendizado de máquina para classificação. O conjunto de dados Iris é um conjunto de dados clássico frequentemente usado para fins de aprendizado de máquina.

## Pré-processamento

Nesta etapa, realizamos o pré-processamento dos dados, que incluiu a coleta, integração e particionamento dos dados.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Coleta e Integração
iris = load_iris()
caracteristicas = iris.data
rotulos = iris.target

# Partição dos dados
carac_treino, carac_teste, rot_treino, rot_teste = train_test_split(caracteristicas, rotulos)
```

# Mineração de Dados
Nesta seção, aplicamos técnicas de aprendizado de máquina para extrair informações úteis dos dados.

## Árvore de Decisão
```python 
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.tree import export_text

    # Criação e treinamento do modelo
    arvore = DecisionTreeClassifier(max_depth=2)
    arvore.fit(X=carac_treino, y=rot_treino)

    # Avaliação do modelo
    rot_preditos = arvore.predict(carac_teste)
    acuracia_arvore = accuracy_score(rot_teste, rot_preditos)

    # Exibição da estrutura da árvore
    r = export_text(arvore, feature_names=iris['feature_names'])
    print("Estrutura da árvore")
    print(r)

    print("Acurácia Árvore de Decisão:", round(acuracia_arvore, 5))
```

## Máquina de Vetor de Suporte (SVM)
```python
    from sklearn.svm import SVC

    # Criação e treinamento do modelo
    clf = SVC()
    clf.fit(X=carac_treino, y=rot_treino)

    # Avaliação do modelo
    rot_preditos_svm = clf.predict(carac_teste)
    acuracia_svm = accuracy_score(rot_teste, rot_preditos_svm)

    print("Acurácia SVM:", round(acuracia_svm, 5))
```
## Pós-processamento
Na etapa final, exibimos os resultados obtidos após a mineração de dados.
```python 
    print("Acurácia Árvore de Decisão:", round(acuracia_arvore, 5))
    print("Acurácia SVM:", round(acuracia_svm, 5))
```
# Conclusão
Este projeto visa não apenas aplicar algoritmos de aprendizado de máquina, mas também entender o processo de análise de dados e como esses algoritmos são utilizados para resolver problemas do mundo real.