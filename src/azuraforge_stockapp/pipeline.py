# app-stock-predictor/src/azuraforge_stockapp/pipeline.py

import logging
from typing import Any, Dict, Tuple, List

import yaml
from importlib import resources
import pandas as pd
import yfinance as yf

from azuraforge_learner import Sequential, LSTM, Linear
from azuraforge_learner.pipelines import TimeSeriesPipeline

# YENİ: Caching modülünü import et
from .caching import get_cache_filepath, load_from_cache, save_to_cache


def get_default_config() -> Dict[str, Any]:
    try:
        with resources.open_text("azuraforge_stockapp.config", "stock_predictor_config.yml") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Hisse senedi uygulaması için varsayılan config yüklenemedi: {e}", exc_info=True)
        return {"error": f"Varsayılan konfigürasyon yüklenemedi: {e}"}

class StockPredictionPipeline(TimeSeriesPipeline):
    """
    Hisse senedi fiyat tahmini için TimeSeriesPipeline'den türetilmiş,
    standartlaştırılmış ve basit pipeline.
    """
    def _load_data(self) -> pd.DataFrame:
        """
        yfinance'ten hisse senedi verilerini çeker.
        Veriyi önce önbellekte arar, bulamazsa API'den çeker ve önbelleğe alır.
        """
        data_sourcing_config = self.config.get("data_sourcing", {})
        system_config = self.config.get("system", {})
        ticker = data_sourcing_config.get("ticker", "MSFT")

        # YENİ: Önbellek mekanizması
        cache_dir = system_config.get("cache_dir", ".cache")
        cache_max_age = system_config.get("cache_max_age_hours", 24)
        # Hangi parametrelerin önbelleği benzersiz kıldığını tanımla
        cache_params = {"ticker": ticker, "period": "max"}
        cache_filepath = get_cache_filepath(cache_dir, "yfinance_data", cache_params)

        cached_data = load_from_cache(cache_filepath, cache_max_age)
        if cached_data is not None:
            return cached_data

        # Önbellekte yoksa veya süresi dolmuşsa, veriyi çek
        self.logger.info(f"Önbellek boş. '{ticker}' için yfinance'ten veri çekiliyor...")
        
        data = yf.download(ticker, period="max", progress=False, actions=False, auto_adjust=True)
        if data.empty:
            raise ValueError(f"'{ticker}' için veri indirilemedi.")
            
        self.logger.info(f"{len(data)} satır veri indirildi.")
        
        # YENİ: Çekilen veriyi önbelleğe kaydet
        save_to_cache(data, cache_filepath)

        return data[['Close']]

    def _get_target_and_feature_cols(self) -> Tuple[str, List[str]]:
        target_col = "Close"
        feature_cols = ["Close"]
        return target_col, feature_cols

    def _create_model(self, input_shape: Tuple) -> Sequential:
        input_size = input_shape[2]
        hidden_size = self.config.get("model_params", {}).get("hidden_size", 50)
        
        return Sequential(
            LSTM(input_size=input_size, hidden_size=hidden_size),
            Linear(hidden_size, 1)
        )