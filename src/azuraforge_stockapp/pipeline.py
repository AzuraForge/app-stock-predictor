# app-stock-predictor/src/azuraforge_stockapp/pipeline.py

import logging
from typing import Any, Dict, Tuple, List

import yaml
from importlib import resources
import pandas as pd
import yfinance as yf

from azuraforge_learner import Sequential, LSTM, Linear
# DÜZELTME: Artık TimeSeriesPipeline'i import ediyoruz
from azuraforge_learner.pipelines import TimeSeriesPipeline

def get_default_config() -> Dict[str, Any]:
    """Uygulamanın varsayılan konfigürasyonunu yükler."""
    try:
        with resources.open_text("azuraforge_stockapp.config", "stock_predictor_config.yml") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Hisse senedi uygulaması için varsayılan config yüklenemedi: {e}", exc_info=True)
        return {"error": f"Varsayılan konfigürasyon yüklenemedi: {e}"}

# DÜZELTME: Artık TimeSeriesPipeline'den miras alıyoruz
class StockPredictionPipeline(TimeSeriesPipeline):
    """
    Hisse senedi fiyat tahmini için TimeSeriesPipeline'den türetilmiş,
    merkezi caching kullanan standartlaştırılmış pipeline.
    """
    
    # YENİ: _load_data_from_source metodunu implemente ediyoruz.
    # Bu metot sadece veriyi kaynağından çekmekle sorumlu. Caching'i düşünmüyor.
    def _load_data_from_source(self) -> pd.DataFrame:
        """Sadece yfinance'ten veri çekme işini yapar."""
        ticker = self.config.get("data_sourcing", {}).get("ticker", "MSFT")
        self.logger.info(f"'{ticker}' için yfinance'ten veri çekiliyor (period=max)...")
        
        data = yf.download(ticker, period="max", progress=False, actions=False, auto_adjust=True)
        if data.empty:
            raise ValueError(f"'{ticker}' için veri indirilemedi.")
            
        self.logger.info(f"{len(data)} satır veri indirildi.")
        return data

    # YENİ: get_caching_params metodunu implemente ediyoruz.
    def get_caching_params(self) -> Dict[str, Any]:
        """
        Önbellek anahtarını oluşturacak benzersiz parametreleri döndürür.
        Hisse senedi tahmini için bu sadece 'ticker'dır.
        """
        return {"ticker": self.config.get("data_sourcing", {}).get("ticker")}

    def _get_target_and_feature_cols(self) -> Tuple[str, List[str]]:
        """Bu basit model için hedef ve özellik aynı sütundur: 'Close'."""
        return "Close", ["Close"]

    def _create_model(self, input_shape: Tuple) -> Sequential:
        """
        LSTM ve bir Linear katmandan oluşan modeli oluşturur.
        input_shape: (num_samples, sequence_length, num_features)
        """
        input_size = input_shape[2] # num_features
        hidden_size = self.config.get("model_params", {}).get("hidden_size", 50)
        
        return Sequential(
            LSTM(input_size=input_size, hidden_size=hidden_size),
            Linear(hidden_size, 1) # Çıktı boyutu her zaman 1 (tek bir değer tahmini)
        )

    # DİKKAT: Artık kendi run() metodumuz yok!
    # Tüm çalıştırma, ölçekleme, eğitim, değerlendirme ve raporlama mantığı,
    # miras aldığımız TimeSeriesPipeline'in run() metodu tarafından hallediliyor.
    # Bu, kodumuzu inanılmaz derecede basitleştiriyor.