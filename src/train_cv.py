import argparse  # Biblioteca para parsing de argumentos de linha de comando.
import os  # Biblioteca para operações no sistema de arquivos.
import pandas as pd  # Biblioteca para manipulação de dados.
import pickle  # Biblioteca para serialização de objetos.
from sklearn.model_selection import StratifiedKFold, GridSearchCV  # Validação cruzada e busca de hiperparâmetros.

from src.preprocess import load_data, preprocess_data  # Funções de pré-processamento.
from src.model import get_model  # Função para obter o modelo.
from src.config import hyperparam_grid  # Grade de hiperparâmetros.
from src.utils import calculate_metrics  # Função para calcular métricas.

def main():
    # Função principal para executar o treinamento com validação cruzada.
    parser = argparse.ArgumentParser()  # Cria o parser de argumentos.
    parser.add_argument('--data', required=True)  # Argumento obrigatório: caminho dos dados.
    parser.add_argument('--out', required=True)  # Argumento obrigatório: diretório de saída para reports.
    parser.add_argument('--k', type=int, default=5)  # Argumento: número de folds (padrão 5).
    parser.add_argument('--seed', type=int, default=42)  # Argumento: semente para reproducibilidade (padrão 42).
    args = parser.parse_args()  # Parseia os argumentos.

    os.makedirs(args.out, exist_ok=True)  # Cria o diretório de saída se não existir.
    df = load_data(args.data)  # Carrega os dados.
    X, y = preprocess_data(df, scale=False)  # Pré-processa os dados (sem escalonamento para árvores).

    skf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=args.seed)  # Cria o validador cruzado estratificado.

    base_model = get_model({}, seed=args.seed)  # Cria modelo base para busca.
    grid_search = GridSearchCV(base_model, hyperparam_grid, cv=skf, scoring='roc_auc', n_jobs=-1)  # Cria GridSearchCV com scoring ROC-AUC.
    grid_search.fit(X, y)  # Executa a busca de hiperparâmetros.

    best_params = grid_search.best_params_  # Obtém os melhores parâmetros.

    # Lista para armazenar métricas de cada fold.
    metrics_list = []
    for train_idx, test_idx in skf.split(X, y):  # Loop sobre os folds.
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]  # Divide features em treino/teste.
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]  # Divide target em treino/teste.
        model = get_model(best_params, seed=args.seed)  # Cria modelo com melhores params.
        model.fit(X_train, y_train)  # Treina o modelo no fold de treino.
        y_pred = model.predict(X_test)  # Prediz no fold de teste.
        y_prob = model.predict_proba(X_test)[:, 1]  # Obtém probabilidades da classe positiva.
        metrics = calculate_metrics(y_test, y_pred, y_prob)  # Calcula métricas.
        metrics_list.append(metrics)  # Adiciona à lista.

    df_metrics = pd.DataFrame(metrics_list)  # Converte lista para DataFrame.
    mean_metrics = df_metrics.mean()  # Calcula média das métricas.
    std_metrics = df_metrics.std()  # Calcula desvio padrão das métricas.
    results = pd.DataFrame({
        'metric': mean_metrics.index,  # Coluna com nomes das métricas.
        'mean': mean_metrics.values,  # Coluna com médias.
        'std': std_metrics.values  # Coluna com desvios padrão.
    })
    results.to_csv(os.path.join(args.out, 'results.csv'), index=False)  # Salva results.csv.

    best_model = get_model(best_params, seed=args.seed)  # Cria o melhor modelo.
    best_model.fit(X, y)  # Treina no dataset completo.
    os.makedirs('artifacts', exist_ok=True)  # Cria diretório artifacts.
    with open('artifacts/best.pkl', 'wb') as f:  # Abre arquivo para escrita binária.
        pickle.dump(best_model, f)  # Salva o modelo serializado.

if __name__ == '__main__':
    # Executa a função main se o script for rodado diretamente.
    main()