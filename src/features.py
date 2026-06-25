"""Feature engineering utilities for the credit risk project."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def add_income_expense_ratio(
    df: pd.DataFrame,
    income_col: str = "income",
    expense_col: str = "expenses",
    output_col: str = "expense_to_income_ratio",
) -> pd.DataFrame:
    """Create an expense-to-income ratio feature when source columns exist."""
    featured = df.copy()
    if income_col in featured.columns and expense_col in featured.columns:
        featured[output_col] = featured[expense_col] / featured[income_col].replace(0, pd.NA)
    return featured


def criar_features_open_finance_simulado(df: pd.DataFrame) -> pd.DataFrame:
    """Create simulated Open Finance-style features from the supervised base."""
    required_columns = {"AMT_INCOME_TOTAL", "AMT_ANNUITY", "TARGET"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise KeyError(f"Missing required columns for Open Finance features: {missing}")

    featured = df.copy()
    featured["RENDA_RECORRENTE_MEDIA_6M"] = featured["AMT_INCOME_TOTAL"] / 12
    featured["COMPROMETIMENTO_RENDA_ESTIMADO"] = featured["AMT_ANNUITY"] / (
        featured["AMT_INCOME_TOTAL"] / 12 + 1
    )
    featured["VOLATILIDADE_SALDO_6M"] = (
        np.random.default_rng(42).normal(0.18, 0.06, len(featured)).clip(0.02, 0.60)
    )
    featured["ATRASOS_PAGAMENTO_12M"] = (
        featured["TARGET"] + np.random.default_rng(7).poisson(0.25, len(featured))
    ).clip(0, 6)
    featured["INDICE_ESTABILIDADE_FINANCEIRA"] = (
        1
        / (1 + featured["COMPROMETIMENTO_RENDA_ESTIMADO"])
        * (1 - featured["VOLATILIDADE_SALDO_6M"])
        * (1 / (1 + featured["ATRASOS_PAGAMENTO_12M"]))
    )
    return featured


def aggregate_bureau_history(bureau: pd.DataFrame) -> pd.DataFrame:
    """Aggregate bureau credit history by current customer id."""
    bureau = bureau.copy()
    bureau["BUREAU_IS_ACTIVE"] = (bureau["CREDIT_ACTIVE"] == "Active").astype(int)
    bureau["BUREAU_IS_CLOSED"] = (bureau["CREDIT_ACTIVE"] == "Closed").astype(int)

    aggregated = bureau.groupby("SK_ID_CURR").agg(
        BUREAU_QTD_CREDITOS=("SK_ID_BUREAU", "count"),
        BUREAU_QTD_CREDITOS_ATIVOS=("BUREAU_IS_ACTIVE", "sum"),
        BUREAU_QTD_CREDITOS_FECHADOS=("BUREAU_IS_CLOSED", "sum"),
        BUREAU_DIAS_CREDITO_MEDIO=("DAYS_CREDIT", "mean"),
        BUREAU_DIAS_CREDITO_MAIS_RECENTE=("DAYS_CREDIT", "max"),
        BUREAU_DIAS_ATUALIZACAO_MEDIO=("DAYS_CREDIT_UPDATE", "mean"),
        BUREAU_ATRASO_DIAS_MAX=("CREDIT_DAY_OVERDUE", "max"),
        BUREAU_ATRASO_DIAS_MEDIO=("CREDIT_DAY_OVERDUE", "mean"),
        BUREAU_VALOR_CREDITO_TOTAL=("AMT_CREDIT_SUM", "sum"),
        BUREAU_VALOR_DIVIDA_TOTAL=("AMT_CREDIT_SUM_DEBT", "sum"),
        BUREAU_VALOR_VENCIDO_TOTAL=("AMT_CREDIT_SUM_OVERDUE", "sum"),
        BUREAU_ANUIDADE_MEDIA=("AMT_ANNUITY", "mean"),
    )
    aggregated["BUREAU_RATIO_DIVIDA_CREDITO"] = (
        aggregated["BUREAU_VALOR_DIVIDA_TOTAL"] / (aggregated["BUREAU_VALOR_CREDITO_TOTAL"] + 1)
    )
    return aggregated.reset_index()


def aggregate_previous_applications(previous: pd.DataFrame) -> pd.DataFrame:
    """Aggregate previous Home Credit applications by current customer id."""
    previous = previous.copy()
    previous["PREV_IS_APPROVED"] = (previous["NAME_CONTRACT_STATUS"] == "Approved").astype(int)
    previous["PREV_IS_REFUSED"] = (previous["NAME_CONTRACT_STATUS"] == "Refused").astype(int)

    aggregated = previous.groupby("SK_ID_CURR").agg(
        PREV_QTD_SOLICITACOES=("SK_ID_PREV", "count"),
        PREV_QTD_APROVADAS=("PREV_IS_APPROVED", "sum"),
        PREV_QTD_RECUSADAS=("PREV_IS_REFUSED", "sum"),
        PREV_VALOR_SOLICITADO_MEDIO=("AMT_APPLICATION", "mean"),
        PREV_VALOR_SOLICITADO_TOTAL=("AMT_APPLICATION", "sum"),
        PREV_VALOR_CREDITO_MEDIO=("AMT_CREDIT", "mean"),
        PREV_VALOR_CREDITO_TOTAL=("AMT_CREDIT", "sum"),
        PREV_ANUIDADE_MEDIA=("AMT_ANNUITY", "mean"),
        PREV_PRAZO_MEDIO=("CNT_PAYMENT", "mean"),
        PREV_DIAS_DECISAO_MEDIO=("DAYS_DECISION", "mean"),
    )
    aggregated["PREV_TAXA_APROVACAO"] = (
        aggregated["PREV_QTD_APROVADAS"] / aggregated["PREV_QTD_SOLICITACOES"].replace(0, pd.NA)
    )
    return aggregated.reset_index()


def aggregate_installments_payments(installments: pd.DataFrame) -> pd.DataFrame:
    """Aggregate installment payment behavior by current customer id."""
    installments = installments.copy()
    installments["INSTALL_DIAS_ATRASO"] = (
        installments["DAYS_ENTRY_PAYMENT"] - installments["DAYS_INSTALMENT"]
    ).clip(lower=0)
    installments["INSTALL_VALOR_PAGO_A_MENOR"] = (
        installments["AMT_INSTALMENT"] - installments["AMT_PAYMENT"]
    ).clip(lower=0)
    installments["INSTALL_IS_ATRASO"] = (installments["INSTALL_DIAS_ATRASO"] > 0).astype(int)

    aggregated = installments.groupby("SK_ID_CURR").agg(
        INSTALL_QTD_PAGAMENTOS=("SK_ID_PREV", "count"),
        INSTALL_VALOR_PARCELAS_TOTAL=("AMT_INSTALMENT", "sum"),
        INSTALL_VALOR_PAGAMENTOS_TOTAL=("AMT_PAYMENT", "sum"),
        INSTALL_VALOR_PARCELA_MEDIO=("AMT_INSTALMENT", "mean"),
        INSTALL_DIAS_ATRASO_MEDIO=("INSTALL_DIAS_ATRASO", "mean"),
        INSTALL_DIAS_ATRASO_MAX=("INSTALL_DIAS_ATRASO", "max"),
        INSTALL_QTD_ATRASOS=("INSTALL_IS_ATRASO", "sum"),
        INSTALL_VALOR_PAGO_A_MENOR_TOTAL=("INSTALL_VALOR_PAGO_A_MENOR", "sum"),
    )
    aggregated["INSTALL_RATIO_ATRASOS"] = (
        aggregated["INSTALL_QTD_ATRASOS"] / aggregated["INSTALL_QTD_PAGAMENTOS"].replace(0, pd.NA)
    )
    aggregated["INSTALL_RATIO_PAGAMENTO"] = (
        aggregated["INSTALL_VALOR_PAGAMENTOS_TOTAL"] / (aggregated["INSTALL_VALOR_PARCELAS_TOTAL"] + 1)
    )
    return aggregated.reset_index()


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
