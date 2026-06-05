import unittest

import pandas as pd

from src.features import criar_features_open_finance_simulado


class OpenFinanceFeatureTest(unittest.TestCase):
    def test_criar_features_open_finance_simulado(self):
        df = pd.DataFrame(
            {
                "AMT_INCOME_TOTAL": [120000.0, 60000.0],
                "AMT_ANNUITY": [12000.0, 6000.0],
                "TARGET": [0, 1],
            }
        )

        featured = criar_features_open_finance_simulado(df)

        self.assertIn("RENDA_RECORRENTE_MEDIA_6M", featured.columns)
        self.assertIn("COMPROMETIMENTO_RENDA_ESTIMADO", featured.columns)
        self.assertIn("VOLATILIDADE_SALDO_6M", featured.columns)
        self.assertIn("ATRASOS_PAGAMENTO_12M", featured.columns)
        self.assertIn("INDICE_ESTABILIDADE_FINANCEIRA", featured.columns)
        self.assertEqual(featured.loc[0, "RENDA_RECORRENTE_MEDIA_6M"], 10000.0)
        self.assertTrue(featured["INDICE_ESTABILIDADE_FINANCEIRA"].between(0, 1).all())


if __name__ == "__main__":
    unittest.main()
