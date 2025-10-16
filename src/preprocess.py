import pandas as pd  # Biblioteca para manipulação de dados.
from sklearn.impute import SimpleImputer  # Imputador para valores ausentes.

def load_data(file_path):
    # Função para carregar o dataset de um arquivo CSV.
    # Parâmetro: file_path - caminho do arquivo CSV.
    df = pd.read_csv(file_path)  # Lê o CSV e armazena em um DataFrame.
    return df  # Retorna o DataFrame carregado.

def preprocess_data(df, scale=False):
    # Função para pré-processar os dados: separa features e target, imputa valores ausentes.
    # Parâmetros: df - DataFrame com dados; scale - se True, aplica escalonamento (não usado para árvores).
    X = df.drop('Potability', axis=1)  # Separa features removendo a coluna target 'Potability'.
    y = df['Potability']  # Extrai a coluna target 'Potability'.
    imputer = SimpleImputer(strategy='median')  # Cria imputador com estratégia de mediana.
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)  # Aplica imputação e converte de volta para DataFrame.
    if scale:
        # Se escalonamento for requerido (não para Decision Tree), aplica StandardScaler.
        from sklearn.preprocessing import StandardScaler  # Importa escalonador.
        scaler = StandardScaler()  # Cria o escalonador.
        X_imputed = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns)  # Aplica escalonamento.
    return X_imputed, y  # Retorna features processadas e target.