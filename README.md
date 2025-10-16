# Water Classifier - Decision Tree

Este repositório contém um projeto completo para um classificador de potabilidade da água usando Árvore de Decisão, baseado no dataset water.csv.

## Descrição
- Modelo: Decision Tree Classifier.
- Dataset: water.csv (alvo: Potability).
- Pré-processamento: Imputação por mediana (sem escalonamento, pois não é necessário para árvores).
- Validação: Stratified K-Fold (k=5 por padrão).
- Busca de hiperparâmetros: Grid Search com métricas como ROC-AUC.
- Métricas: Accuracy, F1, ROC-AUC, Precision, Recall, PR-AUC.
- Figuras: ROC, Precision-Recall, Matriz de Confusão, Importâncias de Features.

## Instalação
1. Clone o repositório.
2. Instale as dependências: `pip install -r requirements.txt`.

## Execução via CLI
- Treinamento e CV: `python -m src.train_cv --data data/raw/water.csv --out reports/ --k 5 --seed 42`
  - Gera `reports/results.csv` e `artifacts/best.pkl`.
- Avaliação e Figuras: `python -m src.evaluate --model artifacts/best.pkl --data data/raw/water.csv --out figures/`
  - Gera figuras em `figures/`.

## Execução via Notebook
- Abra `notebooks/DecisionTree.ipynb` no Jupyter Notebook e execute a única célula para rodar o projeto sequencialmente.

## Reprodutibilidade
- Semente fixa: 42.
- Ambiente: Python 3.12+ com bibliotecas listadas em requirements.txt.

## Licença
MIT.