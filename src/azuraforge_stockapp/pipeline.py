import logging
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os 
from typing import Any
import yaml # YENİ: Konfigürasyon dosyasını okumak için

from azuraforge_learner import Learner, Sequential, Linear, MSELoss, SGD, ReLU

def get_default_config():
    """
    Bu pipeline'ın varsayılan konfigürasyonunu bir Python sözlüğü olarak döndürür.
    YAML dosyasından yükler.
    """
    config_path = os.path.join(os.path.dirname(__file__), "config", "stock_predictor_config.yml")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logging.error(f"Error loading default config for stock_predictor: {e}")
        return {"error": f"Could not load default config: {e}"}

class StockPredictionPipeline:
    def __init__(self, config: dict, celery_task: Any = None):
        # DÜZELTME: Eğer config worker'dan gelmiyorsa (örn. yerel test), varsayılanı yükle
        default_config = get_default_config()
        # Gelen config'i varsayılanın üzerine yaz (merge et)
        self.config = {**default_config, **config}
        
        self.logger = logging.getLogger(self.__class__.__name__)
        logging.basicConfig(level="INFO", format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.celery_task = celery_task

    def run(self):
        data_sourcing_config = self.config.get("data_sourcing", {})
        training_params_config = self.config.get("training_params", {})
        
        ticker = data_sourcing_config.get("ticker", "MSFT")
        start_date = data_sourcing_config.get("start_date", "2021-01-01")
        epochs = training_params_config.get("epochs", 10)
        lr = training_params_config.get("lr", 0.01)

        self.logger.info(f"--- Running Stock Prediction Pipeline for {ticker} ({epochs} epochs, lr={lr}) ---")
        
        try:
            data = yf.download(ticker, start=start_date, progress=False, actions=False, auto_adjust=True)
            if data.empty:
                raise ValueError(f"No data downloaded for ticker: {ticker} from {start_date}")
            self.logger.info(f"Downloaded {len(data)} rows of data.")
            close_prices = data[['Close']].values.astype(np.float32)
        except Exception as e:
            self.logger.error(f"Data download failed for {ticker}: {e}")
            raise 

        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_prices = scaler.fit_transform(close_prices)
        
        if len(scaled_prices) < 2:
            self.logger.warning("Not enough data to create sequences for training.")
            return {"status": "completed", "ticker": ticker, "final_loss": float('inf'), "message": "Not enough data for training"}

        X, y = scaled_prices[:-1], scaled_prices[1:]
        
        model = Sequential(Linear(1, 64), ReLU(), Linear(64, 1))
        criterion = MSELoss()
        optimizer = SGD(model.parameters(), lr=lr)
        
        learner = Learner(model, criterion, optimizer, current_task=self.celery_task)

        self.logger.info(f"Starting training for {epochs} epochs...")
        history = learner.fit(X, y, epochs=epochs)
        
        final_loss = history['loss'][-1]
        self.logger.info(f"Training complete. Final loss: {final_loss:.6f}")
        
        return {"status": "completed", "ticker": ticker, "final_loss": final_loss, "loss": history['loss']}