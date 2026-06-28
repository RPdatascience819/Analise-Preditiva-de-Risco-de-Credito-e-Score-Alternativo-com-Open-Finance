"""Streamlit dashboard para scoring de risco de crédito.

Duas abas:
- Cliente individual: formulário com os campos principais (o resto é imputado).
- Lote (CSV): upload de várias linhas no schema bruto e download dos scores.

Requer os artefatos gerados pelo notebook 03 em ``models/``:
``modelo_credito.pkl``, ``imputers.pkl``, ``encoders.pkl`` e ``scoring_config.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing import tratar_nulos
from src.features import criar_features, codificar_categoricas
from src.scoring import (
    adjust_probability_to_prior,
    probability_to_scorecard_points,
    assign_risk_band,
)

MODELS_DIR = ROOT / "models"
BAND_LABELS = {"low": "Baixo risco", "medium": "Risco médio", "high": "Alto risco"}
BAND_COLORS = {"low": "#2E7D32", "medium": "#E6A100", "high": "#E65100"}

st.set_page_config(page_title="Credit Risk Score", layout="wide")


# --------------------------------------------------------------------------- #
# Artefatos e função de score
# --------------------------------------------------------------------------- #
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODELS_DIR / "modelo_credito.pkl")
    imputers = joblib.load(MODELS_DIR / "imputers.pkl")
    encoders = joblib.load(MODELS_DIR / "encoders.pkl")
    config = json.loads((MODELS_DIR / "scoring_config.json").read_text(encoding="utf-8"))
    return model, imputers, encoders, config


def score_clientes(raw_df: pd.DataFrame, artifacts) -> pd.DataFrame:
    """Aplica o pipeline de produção e devolve PD, score e faixa por linha."""
    model, imputers, encoders, config = artifacts

    df = raw_df.drop(columns=[c for c in ("TARGET", "SK_ID_CURR") if c in raw_df.columns])
    # Garante todas as colunas brutas, na ordem do treino (faltantes viram NaN)
    df = df.reindex(columns=config["raw_feature_order"])
    # Categóricas ausentes -> 'DESCONHECIDO' (preserva dtype object para a codificação)
    for col in encoders:
        if col in df.columns:
            df[col] = df[col].fillna("DESCONHECIDO")

    # Mesmo pipeline do treino (sem refit -> sem leakage)
    df = tratar_nulos(df, fit=False, imputers=imputers)
    df = criar_features(df)
    df = codificar_categoricas(df, fit=False, encoders=encoders)

    pd_bruta = model.predict_proba(df)[:, 1]
    pd_real = adjust_probability_to_prior(pd_bruta, target_prior=config["prior"])
    score = probability_to_scorecard_points(
        pd_real, pdo=config["pdo"], odds_ref=config["odds_ref"], score_ref=config["score_ref"]
    )
    faixa = [
        assign_risk_band(s, medium_min=config["band_medium_min"], low_min=config["band_low_min"])
        for s in np.asarray(score)
    ]
    return pd.DataFrame({"PD": np.asarray(pd_real), "score": np.asarray(score), "faixa": faixa})


def plot_score_gauge(score: int, config: dict):
    """Barra horizontal 300–850 com as zonas de faixa e o marcador do cliente."""
    lo, hi = 300, 850
    m_min, l_min = config["band_medium_min"], config["band_low_min"]
    fig, ax = plt.subplots(figsize=(7, 1.2))
    ax.barh(0, m_min - lo, left=lo, color=BAND_COLORS["high"], alpha=0.85)
    ax.barh(0, l_min - m_min, left=m_min, color=BAND_COLORS["medium"], alpha=0.85)
    ax.barh(0, hi - l_min, left=l_min, color=BAND_COLORS["low"], alpha=0.85)
    ax.axvline(score, color="black", linewidth=3)
    ax.text(score, 0.6, f"{int(score)}", ha="center", fontweight="bold")
    ax.set_xlim(lo, hi)
    ax.set_yticks([])
    ax.set_xlabel("Score (300–850)")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
st.title("Score de Risco de Crédito")
st.caption("Modelo LightGBM + scorecard calibrado (Open Finance). Demonstração de portfólio.")

try:
    artifacts = load_artifacts()
except FileNotFoundError:
    st.error(
        "Artefatos não encontrados em `models/`. Execute o **notebook 03** "
        "para gerar `modelo_credito.pkl`, `imputers.pkl`, `encoders.pkl` e "
        "`scoring_config.json` antes de usar o app."
    )
    st.stop()

_, _, encoders, config = artifacts

with st.sidebar:
    st.header("Faixas de risco")
    st.markdown(
        f"- 🟦 **{BAND_LABELS['low']}**: score ≥ {config['band_low_min']:.0f}\n"
        f"- 🟨 **{BAND_LABELS['medium']}**: {config['band_medium_min']:.0f}–{config['band_low_min']:.0f}\n"
        f"- 🟧 **{BAND_LABELS['high']}**: score < {config['band_medium_min']:.0f}"
    )
    st.caption(
        f"Calibração: prior {config['prior']:.1%} · PDO {config['pdo']} · "
        f"ref {config['score_ref']} @ odds {config['odds_ref']:.1f}:1"
    )


def cat_options(col: str) -> list[str]:
    classes = [c for c in encoders[col].classes_ if c != "DESCONHECIDO"]
    return sorted(classes)


def render_resultado(resultado: pd.Series):
    faixa = resultado["faixa"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Score", f"{int(resultado['score'])}")
    c2.metric("PD (prob. inadimplência)", f"{resultado['PD']:.1%}")
    c3.markdown(
        f"<div style='padding:0.6rem;border-radius:8px;background:{BAND_COLORS[faixa]};"
        f"color:white;text-align:center;font-weight:bold;font-size:1.1rem'>"
        f"{BAND_LABELS[faixa]}</div>",
        unsafe_allow_html=True,
    )
    st.pyplot(plot_score_gauge(int(resultado["score"]), config))


tab_individual, tab_lote = st.tabs(["👤 Cliente individual", "📄 Lote (CSV)"])

# --------------------------- Aba 1: formulário ----------------------------- #
with tab_individual:
    st.subheader("Dados do cliente")
    st.caption("Os campos não preenchidos usam a mediana/categoria padrão do treino.")

    with st.form("form_cliente"):
        col1, col2, col3 = st.columns(3)
        with col1:
            renda = st.number_input("Renda anual (AMT_INCOME_TOTAL)", 0.0, value=150000.0, step=1000.0)
            credito = st.number_input("Valor do crédito (AMT_CREDIT)", 0.0, value=600000.0, step=1000.0)
            anuidade = st.number_input("Anuidade (AMT_ANNUITY)", 0.0, value=27000.0, step=500.0)
            filhos = st.number_input("Nº de filhos (CNT_CHILDREN)", 0, 20, value=0)
        with col2:
            idade = st.slider("Idade (anos)", 18, 90, value=40)
            anos_emprego = st.slider("Anos no emprego atual", 0, 50, value=5)
            genero = st.selectbox("Gênero (CODE_GENDER)", cat_options("CODE_GENDER"))
            contrato = st.selectbox("Tipo de contrato (NAME_CONTRACT_TYPE)", cat_options("NAME_CONTRACT_TYPE"))
        with col3:
            educacao = st.selectbox("Escolaridade (NAME_EDUCATION_TYPE)", cat_options("NAME_EDUCATION_TYPE"))
            estado_civil = st.selectbox("Estado civil (NAME_FAMILY_STATUS)", cat_options("NAME_FAMILY_STATUS"))
            ext2 = st.slider("Score externo 2 (EXT_SOURCE_2)", 0.0, 1.0, value=0.5, step=0.01)
            ext3 = st.slider("Score externo 3 (EXT_SOURCE_3)", 0.0, 1.0, value=0.5, step=0.01)
        submitted = st.form_submit_button("Calcular score", type="primary")

    if submitted:
        linha = {
            "AMT_INCOME_TOTAL": renda,
            "AMT_CREDIT": credito,
            "AMT_ANNUITY": anuidade,
            "CNT_CHILDREN": filhos,
            "DAYS_BIRTH": -idade * 365,
            "DAYS_EMPLOYED": -anos_emprego * 365,
            "CODE_GENDER": genero,
            "NAME_CONTRACT_TYPE": contrato,
            "NAME_EDUCATION_TYPE": educacao,
            "NAME_FAMILY_STATUS": estado_civil,
            "EXT_SOURCE_2": ext2,
            "EXT_SOURCE_3": ext3,
        }
        resultado = score_clientes(pd.DataFrame([linha]), artifacts).iloc[0]
        st.divider()
        render_resultado(resultado)

# --------------------------- Aba 2: lote CSV ------------------------------- #
with tab_lote:
    st.subheader("Scoring em lote")
    st.caption(
        "Envie um CSV com as colunas brutas (schema do `application_train`). "
        "Colunas ausentes são imputadas; `TARGET`/`SK_ID_CURR` são ignoradas."
    )

    template = pd.DataFrame(columns=config["raw_feature_order"])
    st.download_button(
        "Baixar template de colunas (CSV vazio)",
        template.to_csv(index=False).encode("utf-8"),
        file_name="template_scoring.csv",
        mime="text/csv",
    )

    arquivo = st.file_uploader("CSV de clientes", type=["csv"])
    if arquivo is not None:
        entrada = pd.read_csv(arquivo)
        st.write(f"Linhas recebidas: **{len(entrada):,}**")
        with st.spinner("Calculando scores..."):
            resultados = score_clientes(entrada, artifacts)

        saida = entrada.copy()
        saida["PD"] = resultados["PD"].round(4).values
        saida["score"] = resultados["score"].values
        saida["faixa"] = resultados["faixa"].map(BAND_LABELS).values

        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("**Distribuição por faixa**")
            st.dataframe(
                resultados["faixa"].map(BAND_LABELS).value_counts().rename("clientes"),
                use_container_width=True,
            )
        with c2:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.hist(resultados["score"], bins=40, color="#2E5EAA")
            ax.set_xlabel("Score")
            ax.set_ylabel("Clientes")
            ax.set_title("Distribuição dos scores")
            fig.tight_layout()
            st.pyplot(fig)

        st.dataframe(saida.head(50), use_container_width=True)
        st.download_button(
            "Baixar resultados (CSV)",
            saida.to_csv(index=False).encode("utf-8"),
            file_name="scores_resultado.csv",
            mime="text/csv",
        )
