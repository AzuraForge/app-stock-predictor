# ========== DOSYA: app-stock-predictor/src/azuraforge_stockapp/pipeline.py (TAM VE GÜNCELLENMİŞ HALİ) ==========

import logging
from typing import Any, Dict, Tuple, List, Optional

import yaml
from importlib import resources
import pandas as pd
import yfinance as yf
from pydantic import BaseModel
import numpy as np

from azuraforge_learner.pipelines import TimeSeriesPipeline
# Dropout katmanını learner'dan import ediyoruz
from azuraforge_learner import Sequential, LSTM, Linear, Dropout 

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
        super().__init__(config)
        self.logger.info("StockPredictionPipeline (Plugin) initialized successfully.")
    
    def get_config_model(self) -> Optional[type[BaseModel]]:
        """Bu pipeline için Pydantic konfigürasyon modelini döndürür."""
        return StockPredictorConfig
    
    def _load_data_from_source(self) -> pd.DataFrame:
        """
        yfinance'ten veri çeker, veriyi temizler ve zamansal özellikler ekler.
        """
        ticker = self.config.get("data_sourcing", {}).get("ticker", "MSFT")
        self.logger.info(f"'_load_data_from_source' called. Ticker: {ticker}")
        
        # 'auto_adjust=True' genellikle bölünmeleri ve temettüleri ayarlar, bu iyi bir pratiktir.
        data = yf.download(ticker, period="max", progress=False, auto_adjust=True)
        if data.empty:
            raise ValueError(f"No data could be downloaded for ticker '{ticker}'.")
            
        # === VERİ SAĞLAMLAŞTIRMA ADIMLARI ===
        # 1. Sadece sayısal sütunları seç.
        numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
        data = data[numeric_cols].copy()

        # 2. Tüm sütunları float32'ye dönüştür.
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce').astype(np.float32)
        
        # 3. Olası tüm NaN değerlerini doldur.
        data.ffill(inplace=True)
        data.bfill(inplace=True)
        data.fillna(0, inplace=True)
        
        # === YENİ: ZAMAN ÖZELLİKLERİ EKLEME (FEATURE ENGINEERING) ===
        data['day_of_week_sin'] = np.sin(2 * np.pi * data.index.dayofweek / 6.0)
        data['day_of_week_cos'] = np.cos(2 * np.pi * data.index.dayofweek / 6.0)
        data['month_sin'] = np.sin(2 * np.pi * data.index.month / 12.0)
        data['month_cos'] = np.cos(2 * np.pi * data.index.month / 12.0)
        data['day_of_year_sin'] = np.sin(2 * np.pi * data.index.dayofyear / 365.0)
        data['day_of_year_cos'] = np.cos(2 * np.pi * data.index.dayofyear / 365.0)
        self.logger.info("Döngüsel zaman özellikleri eklendi (gün, ay, yıl).")
        # === YENİ ADIM SONU ===

        self.logger.info(f"Downloaded and processed {len(data)} rows of data.")
        return data

    def get_caching_params(self) -> Dict[str, Any]:
        """Önbellek anahtarı için sadece ticker'ın yeterli olduğunu belirtir."""
        ticker = self.config.get("data_sourcing", {}).get("ticker", "MSFT")
        self.logger.info(f"'get_caching_params' called. Ticker: {ticker}")
        return {"ticker": ticker}

    def _get_target_and_feature_cols(self) -> Tuple[str, List[str]]:
        """
        Modelin kullanacağı hedef ve özellik sütunlarını belirler.
        Artık temel OHLCV ve yeni zaman özelliklerini içerir.
        """
        base_features = ["Open", "High", "Low", "Close", "Volume"]
        temporal_features = [
            'day_of_week_sin', 'day_of_week_cos', 
            'month_sin', 'month_cos', 
            'day_of_year_sin', 'day_of_year_cos'
        ]
        all_features = sorted(base_features + temporal_features)
        
        self.logger.info(f"Target: Close, Features: {all_features}")
        return "Close", all_features

    def _create_model(self, input_shape: Tuple) -> Sequential:
        """
        LSTM ve Dropout katmanlarından oluşan, daha sağlam bir modeli oluşturur.
        """
        self.logger.info(f"'_create_model' called with Dropout. Input shape: {input_shape}")
        input_size = input_shape[2] 
        hidden_size = self.config.get("model_params", {}).get("hidden_size", 50)
        
        model = Sequential(
            LSTM(input_size=input_size, hidden_size=hidden_size),
            Dropout(0.2), # Aşırı öğrenmeyi (overfitting) önlemek için
            Linear(hidden_size, 1)
        )
        self.logger.info("LSTM with Dropout model created successfully.")
        return model