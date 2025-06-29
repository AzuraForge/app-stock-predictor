import logging
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os 
from typing import Any
import yaml
from importlib import resources

from azuraforge_learner import Learner, Sequential, Linear, LSTM, MSELoss, SGD, Adam, ReLU

def get_default_config():
    # ... (kod aynı)
    try:
        with resources.open_text("azuraforge_stockapp.config", "stock_predictor_config.yml") as f:
            config = yaml.safe_load(f)
        return config
    except (FileNotFoundError, ModuleNotFoundError, Exception) as e:
        logging.error(f"Error loading default config for stock_predictor: {e}", exc_info=True)
        return {"error": f"Could not load default config: {e}"}

def create_sequences(data, sequence_length):
    # ... (kod aynı)
    xs, ys = [], []
    for i in range(len(data) - sequence_length):
        x = data[i:(i + sequence_length)]
        y = data[i + sequence_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

class StockPredictionPipeline:
    def __init__(self, config: dict, task_id: str): # DÜZELTME: celery_task -> task_id
        # DÜZELTME: Artık celery_task objesini değil, sadece id'sini tutuyoruz
        self.task_id = task_id 
        self.logger = logging.getLogger(self.__class__.__name__)
        logging.basicConfig(level="INFO", format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # ... (config birleştirme mantığı aynı) ...
        default_config = get_default_config()
        if "error" in default_config:
            self.logger.error("Could not load default config. Using only provided config.")
            self.config = config
        else:
            def deep_merge(source: dict, destination: dict) -> dict:
                for key, value in source.items():
                    if isinstance(value, dict) and key in destination and isinstance(destination.get(key), dict):
                        destination[key] = deep_merge(value, destination[key])
                    elif value is not None: 
                        destination[key] = value
                return destination
            
            merged_config = default_config.copy()
            self.config = deep_merge(config, merged_config)

    def run(self):
        # ... (run metodunun başı aynı) ...
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
                raise ValueError(f"No data downloaded for ticker: {ticker} using period='max'")
            self.logger.info(f"Downloaded {len(data)} rows of data.")
            close_prices = data[['Close']].values.astype(np.float32)
        except Exception as e:
            self.logger.error(f"Data download failed for {ticker}: {e}")
            raise 

        # ... (run metodunun ortası aynı) ...
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_prices = scaler.fit_transform(close_prices)
        
        if len(scaled_prices) <= sequence_length:
            self.logger.warning(f"Not enough data to create sequences. Need > {sequence_length}, have {len(scaled_prices)}")
            return {"status": "completed", "ticker": ticker, "final_loss": None, "message": "Not enough data for training"}

        X, y = create_sequences(scaled_prices, sequence_length)
        
        if X.size == 0 or y.size == 0:
            self.logger.warning("Created sequences are empty. Aborting training.")
            return {"status": "completed", "ticker": ticker, "final_loss": None, "message": "Sequence creation resulted in empty data."}

        input_size = X.shape[2] 
        
        model = Sequential(LSTM(input_size=input_size, hidden_size=hidden_size), Linear(hidden_size, 1))
        criterion = MSELoss()
        
        optimizer = Adam(model.parameters(), lr=lr) if optimizer_type == "adam" else SGD(model.parameters(), lr=lr)
        
        # DÜZELTME: Learner'a artık task_id'yi iletiyoruz.
        learner = Learner(model, criterion, optimizer, task_id=self.task_id)

        self.logger.info(f"Starting LSTM training for {epochs} epochs...")
        # DÜZELTME: Learner'a pipeline_name'i de iletiyoruz.
        history = learner.fit(X, y, epochs=epochs, pipeline_name=self.config.get("pipeline_name", "stock_predictor"))
        
        final_loss = history['loss'][-1] if history.get('loss') else None
        self.logger.info(f"Training complete. Final loss: {final_loss}")
        
        return {"status": "completed", "ticker": ticker, "final_loss": final_loss, "loss": history.get('loss', [])}