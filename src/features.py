from sklearn.preprocessing import LabelEncoder


def criar_features(df):
    '''
    Cria variáveis derivadas com significado de negócio.
    Cada feature tem uma justificativa analítica.
    '''
    df = df.copy()
 
    # --- FEATURES DE CAPACIDADE DE PAGAMENTO ---
    # Razão entre anuidade e renda: quanto da renda vai para dívida
    # Quanto maior, maior o risco de inadimplência
    df['RAZAO_ANUIDADE_RENDA'] = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + 1)
 
    # Razão entre crédito solicitado e renda anual
    df['RAZAO_CREDITO_RENDA'] = df['AMT_CREDIT'] / (df['AMT_INCOME_TOTAL'] + 1)
 
    # Prazo implícito do empréstimo (em anos)
    df['PRAZO_ANOS'] = df['AMT_CREDIT'] / (df['AMT_ANNUITY'] + 1) / 12
 
    # --- FEATURES DEMOGRÁFICAS ---
    # Converte dias negativos para idade em anos positiva
    df['IDADE_ANOS'] = -df['DAYS_BIRTH'] / 365
 
    # Tempo de emprego em anos (DAYS_EMPLOYED negativo = está empregado)
    df['EMPREGO_ANOS'] = -df['DAYS_EMPLOYED'] / 365
    # Aposentados têm valor 365243 — corrige para 0
    df.loc[df['EMPREGO_ANOS'] < 0, 'EMPREGO_ANOS'] = 0
 
    # Proporção da vida trabalhada (sinal de estabilidade)
    df['PROP_VIDA_TRABALHADA'] = df['EMPREGO_ANOS'] / (df['IDADE_ANOS'] + 1)
    # Combina os 3 scores externos (média ponderada)
    df['SCORE_EXTERNO_MEDIO'] = df[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']].mean(axis=1)
 
    # Produto dos scores (penaliza quem tem um score muito baixo)
    df['SCORE_EXTERNO_PROD'] = (df['EXT_SOURCE_1'] *
                                df['EXT_SOURCE_2'] *
                                df['EXT_SOURCE_3'])
 
    return df


def codificar_categoricas(df, fit=True, encoders=None):
    '''
    Converte variáveis categóricas em números.
    Usamos Label Encoding para LightGBM (suporta nativamente).
    '''
    df = df.copy()
    cat_cols = df.select_dtypes(include=['object']).columns

    if fit:
        encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        return df, encoders
    else:
        if encoders is None:
            raise ValueError('encoders é obrigatório quando fit=False.')
        for col in cat_cols:
            if col in encoders:
                le = encoders[col]
                # Mapeia categorias conhecidas para seus códigos; categorias
                # não vistas no treino recebem um código próprio (len(classes_)),
                # evitando o ValueError padrão do LabelEncoder.
                mapping = {cls: idx for idx, cls in enumerate(le.classes_)}
                codigo_desconhecido = len(le.classes_)
                df[col] = (
                    df[col].astype(str)
                    .map(mapping)
                    .fillna(codigo_desconhecido)
                    .astype(int)
                )
        return df

