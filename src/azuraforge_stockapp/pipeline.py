# ========== GÜNCELLENECEK DOSYA: app-stock-predictor/src/azuraforge_stockapp/pipeline.py ==========
import logging
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os # Ortam değişkenlerini okumak için

from azuraforge_learner import Learner, Sequential, Linear, MSELoss, SGD, ReLU

class StockPredictionPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        logging.basicConfig(level="INFO", format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    def run(self):
        # Config'den değerleri güvenli bir şekilde al, yoksa varsayılan kullan.
        # Bu, KeyError'ları önler.
        data_sourcing_config = self.config.get("data_sourcing", {})
        training_params_config = self.config.get("training_params", {})
        
        ticker = data_sourcing_config.get("ticker", "MSFT")
        start_date = data_sourcing_config.get("start_date", "2021-01-01")
        epochs = training_params_config.get("epochs", 10)
        lr = training_params_config.get("lr", 0.01)

        self.logger.info(f"--- Running Stock Prediction Pipeline for {ticker} ({epochs} epochs) ---")
        
        # 1. Veri Yükleme
        try:
            data = yf.download(ticker, start=start_date, progress=False, actions=False, auto_adjust=True)
            if data.empty:
                raise ValueError(f"No data downloaded for ticker: {ticker} from {start_date}")
            self.logger.info(f"Downloaded {len(data)} rows of data.")
            close_prices = data[['Close']].values.astype(np.float32)
        except Exception as e:
            self.logger.error(f"Data download failed for {ticker}: {e}")
            raise # Hatayı yukarı fırlat

        # 2. Veri Hazırlama
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_prices = scaler.fit_transform(close_prices)
        
        # Eğer veri çok kısaysa, eğitim yapmadan bitir
        if len(scaled_prices) < 2:
            self.logger.warning("Not enough data to create sequences for training.")
            return {"status": "completed", "ticker": ticker, "final_loss": float('inf'), "message": "Not enough data for training"}

        X, y = scaled_prices[:-1], scaled_prices[1:]
        
        # 3. Model ve Learner'ı Oluşturma
        model = Sequential(Linear(1, 64), ReLU(), Linear(64, 1))
        criterion = MSELoss()
        optimizer = SGD(model.parameters(), lr=lr)
        
        # Learner'a Task objesini ilet (ilerleme raporlamak için)
        learner = Learner(model, criterion, optimizer, current_task=os.getenv("CELERY_TASK_ID", None))

        # 4. Eğitim
        self.logger.info(f"Starting training for {epochs} epochs...")
        history = learner.fit(X, y, epochs=epochs)
        
        final_loss = history['loss'][-1]
        self.logger.info(f"Training complete. Final loss: {final_loss:.6f}")
        
        return {"status": "completed", "ticker": ticker, "final_loss": final_loss}