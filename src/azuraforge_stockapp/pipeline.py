# app-stock-predictor/src/azuraforge_stockapp/pipeline.py

import logging
from typing import Any, Dict, Tuple, List, Optional

import yaml
from importlib import resources
import pandas as pd
import yfinance as yf
from pydantic import BaseModel

from azuraforge_learner.pipelines import TimeSeriesPipeline
from azuraforge_learner import Sequential, LSTM, Linear

from .config_schema import StockPredictorConfig

def get_default_config() -> Dict[str, Any]:
    """Eklentinin varsayılan YAML konfigürasyonunu yükler."""
    try:
        with resources.open_text("azuraforge_stockapp.config", "stock_predictor_config.yml") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Failed to load default config for stock app: {e}", exc_info=True)
        return {"error": f"Default config could not be loaded: {e}"}

class StockPredictionPipeline(TimeSeriesPipeline):
    """
    yfinance kütüphanesini kullanarak hisse senedi verilerini çeker ve LSTM modeli
    ile fiyat tahmini yapar.
    """
    def __init__(self, config: Dict[str, Any]):
        # BasePipeline.__init__ çağrısı burada otomatik olarak yapılır
        # ve self.logger ile self.config'i ayarlar.
        super().__init__(config)
        self.logger.info("StockPredictionPipeline (Plugin) initialized successfully.")
    
    def get_config_model(self) -> Optional[type[BaseModel]]:
        """Bu pipeline için Pydantic konfigürasyon modelini döndürür."""
        return StockPredictorConfig
    
    def _load_data_from_source(self) -> pd.DataFrame:
        """Sadece yfinance'ten veri çekme işini yapar."""
        ticker = self.config.get("data_sourcing", {}).get("ticker", "MSFT")
        self.logger.info(f"'_load_data_from_source' called. Ticker: {ticker}")
        
        data = yf.download(ticker, period="max", progress=False, actions=False, auto_adjust=True)
        if data.empty:
            self.logger.error(f"yfinance returned empty data for ticker '{ticker}'.")
            raise ValueError(f"No data could be downloaded for ticker '{ticker}'.")
            
        self.logger.info(f"Downloaded {len(data)} rows of data.")
        return data

    def get_caching_params(self) -> Dict[str, Any]:
        """Önbellek anahtarı için sadece ticker'ın yeterli olduğunu belirtir."""
        ticker = self.config.get("data_sourcing", {}).get("ticker", "MSFT")
        self.logger.info(f"'get_caching_params' called. Ticker: {ticker}")
        return {"ticker": ticker}

    def _get_target_and_feature_cols(self) -> Tuple[str, List[str]]:
        """Bu basit model için hedef ve özellik aynı sütundur: 'Close'."""
        self.logger.info("'_get_target_and_feature_cols' called. Target: Close")
        return "Close", ["Close"]

    def _create_model(self, input_shape: Tuple) -> Sequential:
        """LSTM ve bir Linear katmandan oluşan modeli oluşturur."""
        self.logger.info(f"'_create_model' called. Input shape: {input_shape}")
        # Girdi şekli (batch, seq_len, features)
        input_size = input_shape[2] 
        hidden_size = self.config.get("model_params", {}).get("hidden_size", 50)
        
        model = Sequential(
            LSTM(input_size=input_size, hidden_size=hidden_size),
            Linear(hidden_size, 1)
        )
        self.logger.info("LSTM model created successfully.")
        return model