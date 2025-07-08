# ========== DOSYA: app-stock-predictor/src/azuraforge_stockapp/pipeline.py ==========
import logging
from typing import Any, Dict, Tuple, List, Optional

import yaml
from importlib import resources
import pandas as pd
import yfinance as yf
from pydantic import BaseModel
import numpy as np

from azuraforge_learner.pipelines import TimeSeriesPipeline
from azuraforge_learner import Sequential, LSTM, Linear

from .config_schema import StockPredictorConfig

def get_default_config() -> Dict[str, Any]:
    try:
        with resources.open_text("azuraforge_stockapp.config", "stock_predictor_config.yml") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Failed to load default config for stock app: {e}", exc_info=True)
        return {"error": f"Default config could not be loaded: {e}"}

class StockPredictionPipeline(TimeSeriesPipeline):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.logger.info("StockPredictionPipeline (Plugin) initialized successfully.")
    
    def get_config_model(self) -> Optional[type[BaseModel]]:
        return StockPredictorConfig
    
    def _load_data_from_source(self) -> pd.DataFrame:
        ticker = self.config.get("data_sourcing", {}).get("ticker", "MSFT")
        self.logger.info(f"'_load_data_from_source' called. Ticker: {ticker}")
        
        data = yf.download(ticker, period="max", progress=False, auto_adjust=True)
        if data.empty:
            raise ValueError(f"No data could be downloaded for ticker '{ticker}'.")
            
        # === YENİ: SAĞLAMLAŞTIRMA ADIMLARI ===
        # 1. Sadece sayısal sütunları seçerek 'Dividends' gibi sütunları dışla.
        numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
        data = data[numeric_cols].copy()

        # 2. Tüm sütunları float32'ye dönüştürmeye zorla.
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce').astype(np.float32)
        
        # 3. Olası tüm NaN değerlerini doldur.
        data.ffill(inplace=True) # Önce ileriye doğru doldur
        data.bfill(inplace=True) # Sonra geriye doğru doldur (başlangıçtaki NaN'lar için)
        data.fillna(0, inplace=True) # Hala varsa 0 ile doldur
        # === SAĞLAMLAŞTIRMA SONU ===

        self.logger.info(f"Downloaded and processed {len(data)} rows of data.")
        return data

    def get_caching_params(self) -> Dict[str, Any]:
        ticker = self.config.get("data_sourcing", {}).get("ticker", "MSFT")
        return {"ticker": ticker}

    def _get_target_and_feature_cols(self) -> Tuple[str, List[str]]:
        # === YENİ: DAHA İYİ ÖZELLİK MÜHENDİSLİĞİ ===
        # Sadece kapanış fiyatı yerine, temel tüm OHLCV verilerini kullanalım.
        # Bu, modelin performansını ciddi şekilde artırma potansiyeline sahiptir.
        self.logger.info("Target: Close, Features: Open, High, Low, Close, Volume")
        return "Close", ["Open", "High", "Low", "Close", "Volume"]

    def _create_model(self, input_shape: Tuple) -> Sequential:
        self.logger.info(f"'_create_model' called. Input shape: {input_shape}")
        input_size = input_shape[2] 
        hidden_size = self.config.get("model_params", {}).get("hidden_size", 50)
        
        model = Sequential(
            LSTM(input_size=input_size, hidden_size=hidden_size),
            Linear(hidden_size, 1)
        )
        self.logger.info("LSTM model created successfully.")
        return model