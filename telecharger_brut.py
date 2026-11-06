"""
Telechargement BRUT des donnees Yahoo Finance - aucun traitement.

    pip install yfinance
    python telecharger_brut.py
"""

import yfinance as yf

TICKERS = ["^GSPC", "^FCHI", "AAPL", "MSFT", "JPM",
           "XOM", "GLD", "^VIX", "EURUSD=X"]

data = yf.download(
    TICKERS,
    start="2010-01-01",
    end="2026-10-24",
    auto_adjust=False,   # on garde tout : Open High Low Close Adj Close Volume
    progress=True,
)

data.to_csv("yahoo_brut.csv")
print(data.shape)
print("Fichier ecrit : yahoo_brut.csv")
