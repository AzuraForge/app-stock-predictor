# app-stock-predictor/src/azuraforge_stockapp/pipeline.py

import logging
from typing import Any, Dict, Tuple, List

import yaml
from importlib import resources
import pandas as pd
import yfinance as yf

from azuraforge_learner import Sequential, LSTM, Linear
from azuraforge_learner.pipelines import TimeSeriesPipeline

def get_default_config() -> Dict[str, Any]:
    """Uygulamanın varsayılan konfigürasyonunu yükler."""
    try:
        with resources.open_text("azuraforge_stockapp.config", "stock_predictor_config.yml") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Hisse senedi uygulaması için varsayılan config yüklenemedi: {e}", exc_info=True)
        return {"error": f"Varsayılan konfigürasyon yüklenemedi: {e}"}

class StockPredictionPipeline(TimeSeriesPipeline):
    """
    Hisse senedi fiyat tahmini için TimeSeriesPipeline'den türetilmiş,
    merkezi caching kullanan standartlaştırılmış pipeline.
    Sadece kendine özgü metotları implemente eder.
    """
    
    def _load_data_from_source(self) -> pd.DataFrame:
        """Sadece yfinance'ten veri çekme işini yapar. Caching bu metodun dışındadır."""
        ticker = self.config.get("data_sourcing", {}).get("ticker", "MSFT")
        self.logger.info(f"'{ticker}' için yfinance'ten veri çekiliyor (period=max)...")
        
        data = yf.download(ticker, period="max", progress=False, actions=False, auto_adjust=True)
        if data.empty:
            raise ValueError(f"'{ticker}' için veri indirilemedi.")
            
        self.logger.info(f"Downloaded {len(data)} rows of data.")
        return data

    def get_caching_params(self) -> Dict[str, Any]:
        """Önbellek anahtarı için sadece ticker'ın yeterli olduğunu belirtir."""
        return {"ticker": self.config.get("data_sourcing", {}).get("ticker", "MSFT")}

    def _get_target_and_feature_cols(self) -> Tuple[str, List[str]]:
        """Bu basit model için hedef ve özellik aynı sütundur: 'Close'."""
        return "Close", ["Close"]

    def _create_model(self, input_shape: Tuple) -> Sequential:
        """LSTM ve bir Linear katmandan oluşan modeli oluşturur."""
        input_size = input_shape[2] # num_features
        hidden_size = self.config.get("model_params", {}).get("hidden_size", 50)
        
        return Sequential(
            LSTM(input_size=input_size, hidden_size=hidden_size),
            Linear(hidden_size, 1)
        )