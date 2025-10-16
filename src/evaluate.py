import argparse  # Biblioteca para parsing de argumentos.
import os  # Biblioteca para operações no sistema de arquivos.
import pickle  # Biblioteca para deserialização de objetos.
import matplotlib.pyplot as plt  # Biblioteca para plotagem.
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve  # Métricas para curvas e matriz.
import seaborn as sns  # Biblioteca para visualizações avançadas.
import pandas as pd  # Biblioteca para manipulação de dados.

from src.preprocess import load_data, preprocess_data  # Funções de pré-processamento.

def main():
    # Função principal para avaliar o modelo e gerar figuras.
    parser = argparse.ArgumentParser()  # Cria o parser de argumentos.
    parser.add_argument('--model', required=True)  # Argumento obrigatório: caminho do modelo pkl.
    parser.add_argument('--data', required=True)  # Argumento obrigatório: caminho dos dados.
    parser.add_argument('--out', required=True)  # Argumento obrigatório: diretório de saída para figures.
    args = parser.parse_args()  # Parseia os argumentos.

    os.makedirs(args.out, exist_ok=True)  # Cria o diretório de saída se não existir.

    with open(args.model, 'rb') as f:  # Abre o arquivo do modelo para leitura binária.
        model = pickle.load(f)  # Carrega o modelo serializado.

    df = load_data(args.data)  # Carrega os dados.
    X, y = preprocess_data(df, scale=False)  # Pré-processa os dados.

    y_pred = model.predict(X)  # Prediz classes usando o modelo (no dataset completo, para demo).
    y_prob = model.predict_proba(X)[:, 1]  # Obtém probabilidades da classe positiva.

    # Matriz de Confusão
    cm = confusion_matrix(y, y_pred)  # Calcula a matriz de confusão.
    plt.figure()  # Cria uma nova figura.
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')  # Plota o heatmap da matriz.
    plt.title('Matriz de Confusão')  # Define título.
    plt.savefig(os.path.join(args.out, 'confusion_matrix.png'))  # Salva a figura.
    plt.close()  # Fecha a figura.

    # Curva ROC
    fpr, tpr, _ = roc_curve(y, y_prob)  # Calcula pontos da curva ROC.
    plt.figure()  # Cria uma nova figura.
    plt.plot(fpr, tpr)  # Plota a curva.
    plt.title('Curva ROC')  # Define título.
    plt.savefig(os.path.join(args.out, 'roc_curve.png'))  # Salva a figura.
    plt.close()  # Fecha a figura.

    # Curva Precision-Recall
    precision, recall, _ = precision_recall_curve(y, y_prob)  # Calcula pontos da curva PR.
    plt.figure()  # Cria uma nova figura.
    plt.plot(recall, precision)  # Plota a curva.
    plt.title('Curva Precision-Recall')  # Define título.
    plt.savefig(os.path.join(args.out, 'pr_curve.png'))  # Salva a figura.
    plt.close()  # Fecha a figura.

    # Importâncias das Features
    importances = model.feature_importances_  # Obtém importâncias das features do modelo.
    features = X.columns  # Obtém nomes das features.
    plt.figure()  # Cria uma nova figura.
    sns.barplot(x=importances, y=features)  # Plota barplot das importâncias.
    plt.title('Importâncias das Features')  # Define título.
    plt.savefig(os.path.join(args.out, 'feature_importances.png'))  # Salva a figura.
    plt.close()  # Fecha a figura.

if __name__ == '__main__':
    # Executa a função main se o script for rodado diretamente.
    main()