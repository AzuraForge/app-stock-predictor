# ========== DOSYA: src/azuraforge_stockapp/pipeline.py ==========
import logging
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from azuraforge_learner import Learner, Sequential, Linear, MSELoss, SGD, ReLU

def get_default_config():
    """Bu pipeline için varsayılan konfigürasyonu döndürür."""
    return {
      "pipeline_name": "stock_predictor",
      "data_sourcing": {
          "ticker": "MSFT", 
          "start_date": "2022-01-01"
      },
      "training_params": {
          "epochs": 10, 
          "lr": 0.01
      }
    }

class StockPredictionPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        logging.basicConfig(level="INFO", format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    def run(self):
        ticker = self.config.get("data_sourcing", {}).get("ticker")
        self.logger.info(f"--- Running Stock Prediction Pipeline for {ticker} ---")
        
        # 1. Veri Yükleme
        data = yf.download(ticker, start=self.config['data_sourcing']['start_date'], progress=False)
        self.logger.info(f"Downloaded {len(data)} rows of data.")
        close_prices = data[['Close']].values.astype(np.float32)

        # 2. Veri Hazırlama
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_prices = scaler.fit_transform(close_prices)
        X, y = scaled_prices[:-1], scaled_prices[1:]
        
        # 3. Model ve Learner'ı Oluşturma
        model = Sequential(Linear(1, 64), ReLU(), Linear(64, 1))
        criterion = MSELoss()
        optimizer = SGD(model.parameters(), lr=self.config.get("training_params", {}).get("lr", 0.01))
        learner = Learner(model, criterion, optimizer)

        # 4. Eğitim
        self.logger.info(f"Starting training for {self.config['training_params']['epochs']} epochs...")
        history = learner.fit(X, y, epochs=self.config['training_params']['epochs'])
        
        final_loss = history['loss'][-1]
        self.logger.info(f"Training complete. Final loss: {final_loss:.6f}")
        
        return {"status": "completed", "ticker": ticker, "final_loss": final_loss}