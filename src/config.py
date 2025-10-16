# Configurações globais e hiperparâmetros para o modelo.

# Dicionário de grade de hiperparâmetros para GridSearchCV.
hyperparam_grid = {
    "criterion": ["gini", "entropy"],  # Critério de divisão: gini ou entropy.
    "max_depth": [None, 5, 10, 15],  # Profundidade máxima da árvore: None para ilimitada.
    "min_samples_split": [2, 5, 10],  # Mínimo de amostras para dividir um nó.
    "min_samples_leaf": [1, 2, 4],  # Mínimo de amostras em uma folha.
}