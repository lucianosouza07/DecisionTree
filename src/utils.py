from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             precision_score, recall_score, average_precision_score)  # Métricas de avaliação.

def calculate_metrics(y_true, y_pred, y_prob):
    # Função para calcular múltiplas métricas de desempenho.
    # Parâmetros: y_true - rótulos verdadeiros; y_pred - predições; y_prob - probabilidades da classe positiva.
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),  # Calcula acurácia.
        'f1': f1_score(y_true, y_pred),  # Calcula F1-score.
        'roc_auc': roc_auc_score(y_true, y_prob),  # Calcula ROC-AUC.
        'precision': precision_score(y_true, y_pred),  # Calcula precisão.
        'recall': recall_score(y_true, y_pred),  # Calcula recall.
        'pr_auc': average_precision_score(y_true, y_prob)  # Calcula PR-AUC.
    }
    return metrics  # Retorna dicionário com as métricas.