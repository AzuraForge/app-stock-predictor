# app-stock-predictor/src/azuraforge_stockapp/pipeline.py

import logging
from typing import Any, Dict, Tuple, Optional, List
import yaml
from importlib import resources

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

from azuraforge_learner import (
    BasePipeline, Learner, Sequential, LSTM, Linear, Adam, SGD, Callback, MSELoss
)

def create_sequences(data: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """Veriyi (samples, sequence_length, features) formatına dönüştürür."""
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def get_default_config() -> Dict[str, Any]:
    try:
        with resources.open_text("azuraforge_stockapp.config", "stock_predictor_config.yml") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Error loading default config for stock_predictor: {e}", exc_info=True)
        return {"error": f"Could not load default config: {e}"}

class StockPredictionPipeline(BasePipeline):
    """
    Hisse senedi fiyat tahmini için BasePipeline'den türetilmiş,
    standartlaştırılmış pipeline.
    """
    def _load_data(self) -> pd.DataFrame:
        """yfinance'ten hisse senedi verilerini çeker."""
        ticker = self.config.get("data_sourcing", {}).get("ticker", "MSFT")
        self.logger.info(f"'{ticker}' için yfinance'ten veri çekiliyor (period=max)...")
        try:
            data = yf.download(ticker, period="max", progress=False, actions=False, auto_adjust=True)
            if data.empty:
                raise ValueError(f"No data downloaded for ticker: {ticker}")
            self.logger.info(f"Downloaded {len(data)} rows of data.")
            return data[['Close']]
        except Exception as e:
            self.logger.error(f"Data download failed for {ticker}: {e}")
            raise

    def _preprocess(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Veriyi ön işler: ölçekler ve LSTM için sekanslara ayırır.
        BasePipeline'in X ve y beklemesine rağmen, LSTM için bu adımda
        sekans oluşturma daha mantıklıdır. Bu nedenle X, y'yi burada oluşturup döndürüyoruz.
        """
        seq_length = self.config.get("model_params", {}).get("sequence_length", 60)
        
        # BasePipeline'in scaler'ını kullanmak yerine, burada kendi scaler'ımızı yönetiyoruz.
        # Çünkü tüm veri setini tek seferde ölçekleyip sonra sekanslara ayırmalıyız.
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_data = self.scaler.fit_transform(data)

        if len(scaled_data) <= seq_length:
            self.logger.warning(f"Not enough data for sequences. Need > {seq_length}, have {len(scaled_data)}")
            return np.array([]), np.array([])
        
        X, y = create_sequences(scaled_data, seq_length)
        return X, y

    def _create_model(self, input_shape: Tuple) -> Sequential:
        """LSTM tabanlı modeli oluşturur."""
        # input_shape: (num_samples, sequence_length, num_features)
        input_size = input_shape[2] # num_features
        hidden_size = self.config.get("model_params", {}).get("hidden_size", 50)
        
        return Sequential(
            LSTM(input_size=input_size, hidden_size=hidden_size),
            Linear(hidden_size, 1)
        )

    def _create_learner(self, model: Sequential, callbacks: Optional[List[Callback]]) -> Learner:
        """Adam veya SGD optimizer ile Learner'ı oluşturur."""
        training_params = self.config.get("training_params", {})
        lr = training_params.get("lr", 0.001)
        optimizer_type = training_params.get("optimizer", "adam").lower()

        if optimizer_type == "adam":
            optimizer = Adam(model.parameters(), lr=lr)
        else:
            optimizer = SGD(model.parameters(), lr=lr)
            
        return Learner(model, MSELoss(), optimizer, callbacks=callbacks)

    def run(self, callbacks: Optional[List[Callback]] = None) -> Dict[str, Any]:
        """
        Deneyin tam yaşam döngüsünü çalıştırır. BasePipeline'in run metodunu
        zaman serisine özel mantıkla override ediyoruz.
        """
        self.logger.info("Stock Prediction Pipeline başlatılıyor...")
        
        raw_data = self._load_data()
        X, y = self._preprocess(raw_data)

        if X.size == 0:
            return {"status": "failed", "message": "Preprocessing resulted in empty dataset."}

        # Veriyi burada bölüyoruz, çünkü sekanslar zaten oluşturuldu.
        test_size = self.config.get("training_params", {}).get("test_size", 0.2)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Test setindeki verilerin zaman damgalarını al
        time_index_test = raw_data.index[len(raw_data) - len(X_test):]

        model = self._create_model(X_train.shape)
        learner = self._create_learner(model, callbacks)

        epochs = self.config.get("training_params", {}).get("epochs", 50)
        self.logger.info(f"{epochs} epoch için model eğitimi başlıyor...")
        history = learner.fit(X_train, y_train, epochs=epochs, pipeline_name=self.config.get("pipeline_name"))
        
        self.logger.info("Model değerlendiriliyor...")
        y_pred_scaled = learner.predict(X_test)
        
        # Ölçeklemeyi geri al
        y_pred_unscaled = self.scaler.inverse_transform(y_pred_scaled)
        y_test_unscaled = self.scaler.inverse_transform(y_test)

        from sklearn.metrics import r2_score, mean_absolute_error
        metrics = {
            'r2_score': r2_score(y_test_unscaled, y_pred_unscaled),
            'mae': mean_absolute_error(y_test_unscaled, y_pred_unscaled)
        }
        
        final_results = {
            "history": history,
            "metrics": metrics,
            "y_true": y_test_unscaled,
            "y_pred": y_pred_unscaled,
            "time_index": time_index_test,
            "y_label": "Hisse Fiyatı"
        }

        self.logger.info("Rapor oluşturuluyor...")
        # Bu fonksiyon artık learner kütüphanesinden gelecek
        from azuraforge_learner.utils import generate_regression_report
        generate_regression_report(final_results, self.config)
        
        return {
            "final_loss": history['loss'][-1] if history.get('loss') else None,
            "metrics": metrics
        }