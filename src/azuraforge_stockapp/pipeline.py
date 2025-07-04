# app-stock-predictor/src/azuraforge_stockapp/pipeline.py

import logging
from typing import Any, Dict, Tuple, List

import yaml
from importlib import resources
import pandas as pd
import yfinance as yf

from azuraforge_learner.pipelines import TimeSeriesPipeline
from azuraforge_learner import Sequential, LSTM, Linear

from .config_schema import StockPredictorConfig
from pydantic import BaseModel

def get_default_config() -> Dict[str, Any]:
    try:
        with resources.open_text("azuraforge_stockapp.config", "stock_predictor_config.yml") as f:
            return yaml.safe_load(f)
    except Exception as e:
        # Bu log, worker'ın ana sürecinde görünmeli
        logging.error(f"Hisse senedi uygulaması için varsayılan config yüklenemedi: {e}", exc_info=True)
        return {"error": f"Varsayılan konfigürasyon yüklenemedi: {e}"}

class StockPredictionPipeline(TimeSeriesPipeline):
    def get_config_model(self) -> type[BaseModel]:
        """Bu pipeline için Pydantic konfigürasyon modelini döndürür."""
        return StockPredictorConfig
    
    # DÜZELTME: __init__ metodunu, temel sınıfı çağırmak ve loglamayı doğrulamak için geri ekliyoruz.
    def __init__(self, config: Dict[str, Any]):
        # Bu, BasePipeline'in __init__'ini çağıracak ve self.logger'ı ayarlayacaktır.
        super().__init__(config)
        self.logger.info(f"StockPredictionPipeline (Eklenti) başarıyla başlatıldı.")

    def _load_data_from_source(self) -> pd.DataFrame:
        """Sadece yfinance'ten veri çekme işini yapar. Caching bu metodun dışındadır."""
        ticker = self.config.get("data_sourcing", {}).get("ticker", "MSFT")
        self.logger.info(f"'_load_data_from_source' çağrıldı. Ticker: {ticker}")
        
        data = yf.download(ticker, period="max", progress=False, actions=False, auto_adjust=True)
        if data.empty:
            self.logger.error(f"'{ticker}' için yfinance'ten boş veri döndü.")
            raise ValueError(f"'{ticker}' için veri indirilemedi.")
            
        self.logger.info(f"{len(data)} satır veri indirildi.")
        return data

    def get_caching_params(self) -> Dict[str, Any]:
        """Önbellek anahtarı için sadece ticker'ın yeterli olduğunu belirtir."""
        ticker = self.config.get("data_sourcing", {}).get("ticker", "MSFT")
        self.logger.info(f"'get_caching_params' çağrıldı. Ticker: {ticker}")
        return {"ticker": ticker}

    def _get_target_and_feature_cols(self) -> Tuple[str, List[str]]:
        """Bu basit model için hedef ve özellik aynı sütundur: 'Close'."""
        self.logger.info("'_get_target_and_feature_cols' çağrıldı. Hedef: Close")
        return "Close", ["Close"]

    def _create_model(self, input_shape: Tuple) -> Sequential:
        """LSTM ve bir Linear katmandan oluşan modeli oluşturur."""
        self.logger.info(f"'_create_model' çağrıldı. Girdi şekli: {input_shape}")
        input_size = input_shape[2] 
        hidden_size = self.config.get("model_params", {}).get("hidden_size", 50)
        
        model = Sequential(
            LSTM(input_size=input_size, hidden_size=hidden_size),
            Linear(hidden_size, 1)
        )
        self.logger.info("LSTM modeli başarıyla oluşturuldu.")
        return model
