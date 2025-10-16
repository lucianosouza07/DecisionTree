from sklearn.tree import DecisionTreeClassifier  # Classificador de Árvore de Decisão.

def get_model(params, seed=42):
    # Função para criar o modelo de Árvore de Decisão com parâmetros dados.
    # Parâmetros: params - dicionário de hiperparâmetros; seed - semente para reproducibilidade.
    model = DecisionTreeClassifier(random_state=seed, **params)  # Instancia o modelo com semente e params.
    return model  # Retorna o modelo criado.