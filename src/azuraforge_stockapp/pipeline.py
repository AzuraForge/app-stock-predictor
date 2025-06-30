# app-stock-predictor/src/azuraforge_stockapp/pipeline.py

import logging
from typing import Any, Dict, Tuple, Optional, List
import yaml
from importlib import resources

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

from azuraforge_learner import Learner, Sequential, LSTM, Linear, Adam, SGD, MSELoss, Callback
from azuraforge_learner.reporting import generate_regression_report


def create_sequences(data: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
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

class StockPredictionPipeline:
    def __init__(self, config: dict):
        self.logger = logging.getLogger(self.__class__.__name__)
        logging.basicConfig(level="INFO", format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        default_config = get_default_config()
        if "error" in default_config:
            self.config = config
        else:
            def deep_merge(source: dict, destination: dict) -> dict:
                for key, value in source.items():
                    if isinstance(value, dict) and key in destination and isinstance(destination.get(key), dict):
                        destination[key] = deep_merge(value, destination[key])
                    elif value is not None: 
                        destination[key] = value
                return destination
            self.config = deep_merge(config, default_config.copy())
        
        self.scaler = MinMaxScaler(feature_range=(-1, 1))

    def run(self, callbacks: Optional[List[Callback]] = None):
        data_sourcing_config = self.config.get("data_sourcing", {})
        training_params_config = self.config.get("training_params", {})
        model_params_config = self.config.get("model_params", {})
        
        ticker = data_sourcing_config.get("ticker", "MSFT")
        epochs = int(training_params_config.get("epochs", 50)) 
        lr = float(training_params_config.get("lr", 0.001)) 
        optimizer_type = str(training_params_config.get("optimizer", "adam")).lower()
        sequence_length = int(model_params_config.get("sequence_length", 60))
        hidden_size = int(model_params_config.get("hidden_size", 50))

        self.logger.info(f"--- Running LSTM Stock Prediction for {ticker} (period=max, epochs={epochs}) ---")
        
        try:
            data = yf.download(ticker, period="max", progress=False, actions=False, auto_adjust=True)
            if data.empty:
                raise ValueError(f"No data downloaded for ticker: {ticker}")
            self.logger.info(f"Downloaded {len(data)} rows of data.")
            close_prices_df = data[['Close']]
        except Exception as e:
            self.logger.error(f"Data download failed for {ticker}: {e}")
            raise 

        scaled_prices = self.scaler.fit_transform(close_prices_df)
        
        if len(scaled_prices) <= sequence_length:
            self.logger.warning(f"Not enough data for sequences.")
            return {"status": "failed", "message": "Not enough data for training"}
        
        X, y = create_sequences(scaled_prices, sequence_length)
        
        if X.size == 0:
            return {"status": "failed", "message": "Preprocessing resulted in empty dataset."}

        test_size = training_params_config.get("test_size", 0.2)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        time_index_test = close_prices_df.index[split_idx + sequence_length:]

        input_size = X_train.shape[2]
        model = Sequential(LSTM(input_size=input_size, hidden_size=hidden_size), Linear(hidden_size, 1))
        criterion = MSELoss()
        optimizer = Adam(model.parameters(), lr=lr) if optimizer_type == "adam" else SGD(model.parameters(), lr=lr)
        
        learner = Learner(model, criterion, optimizer, callbacks=callbacks)

        self.logger.info(f"{epochs} epoch için model eğitimi başlıyor...")
        history = learner.fit(X_train, y_train, epochs=epochs, pipeline_name=self.config.get("pipeline_name"))
        
        self.logger.info("Model değerlendiriliyor...")
        y_pred_scaled = learner.predict(X_test)
        
        y_pred_unscaled = self.scaler.inverse_transform(y_pred_scaled)
        y_test_unscaled = self.scaler.inverse_transform(y_test)

        from sklearn.metrics import r2_score, mean_absolute_error
        metrics = {
            'r2_score': r2_score(y_test_unscaled, y_pred_unscaled),
            'mae': mean_absolute_error(y_test_unscaled, y_pred_unscaled)
        }
        
        final_results = {
            "history": history, "metrics": metrics,
            "y_true": y_test_unscaled, "y_pred": y_pred_unscaled,
            "time_index": time_index_test, "y_label": "Hisse Fiyatı"
        }

        self.logger.info("Rapor oluşturuluyor...")
        generate_regression_report(final_results, self.config)
        
        return {
            "final_loss": history['loss'][-1] if history.get('loss') else None,
            "metrics": metrics
        }