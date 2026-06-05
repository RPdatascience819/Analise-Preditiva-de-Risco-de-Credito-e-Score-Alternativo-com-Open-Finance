# Analise Preditiva de Risco de Credito e Score Alternativo com Open Finance

Pipeline de risco de credito usando dados financeiros, engenharia de features e um score alternativo inspirado em Open Finance.

## Estrutura

```text
data/raw/          Dados originais baixados
data/processed/    Dados tratados
notebooks/         Analises, features, modelagem e scorecard
src/               Funcoes reutilizaveis
models/            Modelos salvos
dashboard/         Aplicacao Streamlit
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

## Kaggle

A autenticacao local usa um token salvo em `.kaggle/access_token`, pasta ignorada pelo Git.

Para chamar a API pelo ambiente do projeto:

```powershell
.\scripts\kaggle.ps1 datasets list --search credit --page 1
```
