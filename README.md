# Análise Preditiva de Risco de Crédito e Score Alternativo com Open Finance

[![CI](https://github.com/RPdatascience819/Analise-Preditiva-de-Risco-de-Credito-e-Score-Alternativo-com-Open-Finance/actions/workflows/ci.yml/badge.svg)](https://github.com/RPdatascience819/Analise-Preditiva-de-Risco-de-Credito-e-Score-Alternativo-com-Open-Finance/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12-blue)

Pipeline ponta a ponta de **risco de crédito**: do dado bruto ao **score acionável (300–850)**, com pré-processamento sem *data leakage*, modelo (LightGBM), interpretabilidade (SHAP), scorecard calibrado, monitoramento de *drift* e *fairness*, e um app interativo de scoring.

> Base: [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) (`TARGET`: `0 = pagou`, `1 = inadimpliu`; taxa real de inadimplência ≈ 8%). O diferencial do projeto é simular **features de Open Finance** (renda recorrente, comprometimento, volatilidade de saldo) sobre essa base.

## Problema de negócio

Conceder crédito é equilibrar **risco × inclusão**: recusar bons pagadores custa receita; aprovar maus pagadores gera perda. O objetivo é estimar a **probabilidade de inadimplência (PD)** de cada cliente, traduzi-la em um **score** e em **faixas de decisão** (aprovar / analisar / recusar), de forma **interpretável, monitorável e auditável** — requisitos de um modelo de crédito real e regulado.

## Destaques técnicos

- **Sem data leakage:** imputação, engenharia de features e codificação aprendem **apenas no treino** (padrão `fit`/`transform`) e são reaplicadas ao teste/produção.
- **Desbalanceamento tratado corretamente:** `SMOTENC` (ciente de variáveis categóricas) aplicado **dentro de cada fold** da validação cruzada, nunca no teste.
- **Tuning** de hiperparâmetros com **Optuna**; métricas de crédito (AUC, KS, Gini, curva de Lorenz).
- **Interpretabilidade** com SHAP (importância global, *beeswarm* e *waterfall* individual).
- **Scorecard calibrado:** correção de *prior* (remove o viés do treino balanceado) + método **PDO/log-odds** + faixas de risco *data-driven*.
- **Monitoramento e fairness:** PSI (*drift*), estabilidade de faixas e *Disparate Impact* por grupo.
- **Engenharia de software:** módulos reutilizáveis em `src/`, testes (`unittest`), **CI no GitHub Actions** e *branch protection* na `main`.

## Pipeline (notebooks)

| # | Notebook | O que faz |
|---|----------|-----------|
| 01 | `01_eda.ipynb` | Análise exploratória: alvo, variáveis numéricas e categóricas vs. inadimplência. |
| 02 | `02_feature_engineering.ipynb` | Demonstra as transformações (`src/`): nulos, features de negócio, Open Finance simulado, codificação. |
| 03 | `03_model_training.ipynb` | Pré-processamento sem leakage → SMOTENC → Optuna → modelo final + métricas; salva os artefatos de deploy. |
| 04 | `04_scorecard.ipynb` | Regressão Logística interpretável sobre top-features SHAP; conversão PD → pontos (PDO) + faixas de risco. |
| 05 | `05_monitoring_fairness.ipynb` | PSI/*drift*, estabilidade das faixas e métricas de *fairness* por gênero. |

## Resultados

Modelo campeão (LightGBM) no conjunto de teste (~61,5 mil clientes, nunca vistos):

| Métrica | Valor |
|---------|-------|
| AUC-ROC | ≈ 0,76 |
| Gini | ≈ 0,52 |
| KS | ≈ 0,39 |

Faixas de risco do scorecard (inadimplência observada, monotônica):

| Faixa | Inadimplência | Leitura de negócio |
|-------|---------------|--------------------|
| 🟦 Baixo risco | ≈ 2% | Aprovação automática / melhores condições |
| 🟨 Risco médio | ≈ 6% | Aprovar com análise / limite ajustado |
| 🟧 Alto risco | ≈ 17% | Revisão manual / garantias / recusa |

*Fairness:* *Disparate Impact ratio* ≈ 0,80 (no limite da "regra dos 80%") — sinalizado e discutido no notebook 05. *Drift:* PSI ≈ 0 entre treino e teste (estável).

## App de scoring (Streamlit)

`dashboard/app.py` carrega os artefatos e oferece duas abas:

- **👤 Cliente individual** — formulário com os campos principais (o restante é imputado); retorna **PD, score (300–850) e faixa** com *gauge* visual.
- **📄 Lote (CSV)** — *upload* de clientes no schema bruto, scoring em lote, distribuição por faixa e *download* dos resultados.

```bash
make dashboard          # ou: streamlit run dashboard/app.py
```

> O app requer os artefatos em `models/` (gerados ao executar o notebook 03): `modelo_credito.pkl`, `imputers.pkl`, `encoders.pkl`, `scoring_config.json`.

## Estrutura do repositório

```text
data/raw/          Dados originais (não versionados)
data/interim/      Bases intermediárias
data/processed/    Dataset modelável final
notebooks/         EDA, features, modelagem, scorecard e monitoramento (01–05)
src/               Módulos reutilizáveis (preprocessing, features, scoring, monitoring, train, ...)
dashboard/         Aplicação Streamlit
models/            Artefatos de modelo/scoring (não versionados)
reports/figures/   Figuras geradas pelas análises
tests/             Testes automatizados (unittest)
.github/workflows/ CI (GitHub Actions)
```

## Como rodar

Pré-requisitos: Python 3.12. Atalhos disponíveis no `Makefile`.

```bash
# 1. Ambiente
python -m venv venv
# Windows:  .\venv\Scripts\Activate.ps1   |  Linux/macOS: source venv/bin/activate
make install                      # pip install -r requirements.txt

# 2. Dados: baixe o dataset Home Credit do Kaggle em data/raw/

# 3. Notebooks (gera os artefatos em models/)
make notebook                     # execute 01 -> 05 na ordem

# 4. App
make dashboard

# 5. Testes
make test                         # python -m unittest discover -s tests
```

Ou via **Docker**:

```bash
docker build -t credit-score .
docker run -p 8501:8501 credit-score
```

### Base enriquecida (opcional)

```bash
make dataset                      # python -m src.build_dataset
```

Agrega histórico de `bureau.csv`, `previous_application.csv` e `installments_payments.csv` à base supervisionada `application_train.csv`.

## Qualidade

- **Testes** automatizados (`tests/`) cobrindo features, scoring, monitoramento, treino e estrutura.
- **CI** (GitHub Actions) roda a suíte em cada *push*/PR; a `main` é protegida (PR + CI verde obrigatórios).

## Kaggle

A autenticação local usa um token em `.kaggle/access_token` (ignorado pelo Git):

```powershell
.\scripts\kaggle.ps1 datasets list --search credit --page 1
```

## Aviso sobre as features de Open Finance

As variáveis de Open Finance (`criar_features_open_finance_simulado`) são **simuladas** para ilustrar o conceito — não há dados transacionais reais disponíveis publicamente. Elas demonstram a *forma* dessas features; o pipeline de treino sem leakage (notebook 03) usa apenas features derivadas legítimas. Em produção, viriam de extratos/transações via Open Finance.
