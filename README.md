# Analise Preditiva de Risco de Credito e Score Alternativo com Open Finance

Pipeline de risco de credito usando dados financeiros, engenharia de features e um score alternativo inspirado em Open Finance.

## Estrutura

```text
data/raw/                  Dados originais, nunca editados
data/interim/              Bases intermediarias
data/processed/            Dataset modelavel final
notebooks/                 Analises, features, modelagem, scorecard e monitoramento
src/                       Modulos reutilizaveis da pipeline
dashboard/                 Aplicacao Streamlit
models/                    Modelos salvos
reports/figures/           Figuras e artefatos visuais de analise
tests/                     Testes automatizados
```

## Como usar

```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter notebook
```

Para executar o dashboard:

```powershell
streamlit run dashboard\app.py
```

Para rodar os testes:

```powershell
python -m unittest discover -s tests
```

## Kaggle

A autenticacao local usa um token salvo em `.kaggle/access_token`, pasta ignorada pelo Git.

Para chamar a API pelo ambiente do projeto:

```powershell
.\scripts\kaggle.ps1 datasets list --search credit --page 1
```
