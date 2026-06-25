

import pandas as pd
import numpy as np
 
CAT_FILL = 'DESCONHECIDO'


def tratar_nulos(df, fit=True, imputers=None):
    '''
    Trata valores nulos de forma adequada ao contexto de crédito.
    REGRA: nunca simplesmente deleta linhas com nulos — isso gera viés.

    Padrão fit/transform para evitar data leakage:
    - fit=True  : aprende as medianas no conjunto de treino e devolve (df, imputers).
    - fit=False : reaplica as medianas aprendidas (passar imputers) e devolve df.
    '''
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=['object']).columns

    if fit:
        # Medianas aprendidas apenas neste conjunto (robustas a outliers)
        medianas = {col: df[col].median() for col in num_cols}
        imputers = {'medianas': medianas, 'cat_fill': CAT_FILL}
    elif imputers is None:
        raise ValueError('imputers é obrigatório quando fit=False.')

    # Variáveis numéricas: preenche com a mediana aprendida
    for col in num_cols:
        if col in imputers['medianas']:
            df[col] = df[col].fillna(imputers['medianas'][col])

    # Variáveis categóricas: preenche com 'DESCONHECIDO'
    for col in cat_cols:
        df[col] = df[col].fillna(imputers['cat_fill'])

    return (df, imputers) if fit else df
