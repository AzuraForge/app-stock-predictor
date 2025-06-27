# ========== DOSYA: src/azuraforge_stockapp/pipeline.py ==========
import logging
import yfinance as yf
import pandas as pd

# Kütüphanelerimizi import ediyoruz
from azuraforge_learner import Learner, Sequential, Linear, MSELoss, SGD

class StockPredictionPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        logging.basicConfig(level="INFO")

    def run(self):
        self.logger.info(f"Running pipeline: {self.config.get('pipeline_name')}...")
        ticker = self.config.get("data_sourcing", {}).get("ticker")
        data = yf.download(ticker, start=self.config['data_sourcing']['start_date'], progress=False)
        self.logger.info(f"Downloaded {len(data)} rows for {ticker}")
        
        # Burada gerçek eğitim mantığı olacak...
        self.logger.info("Simulating training...")
        
        return {"status": "completed", "ticker": ticker, "final_loss": 0.05}