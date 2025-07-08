from pydantic import BaseModel, Field
from typing import Literal

class DataSourcingConfig(BaseModel):
    ticker: str = Field(..., description="The stock ticker symbol, e.g., 'MSFT'.")

class FeatureEngineeringConfig(BaseModel):
    target_col_transform: Literal['log', 'none'] = 'log'

class ModelParamsConfig(BaseModel):
    sequence_length: int = Field(60, gt=0, description="Length of the input sequence.")
    hidden_size: int = Field(50, gt=0, description="Number of features in the LSTM hidden state.")

class TrainingParamsConfig(BaseModel):
    epochs: int = Field(50, gt=0)
    lr: float = Field(0.001, gt=0, description="Learning rate.")
    optimizer: Literal['adam', 'sgd'] = 'adam'
    test_size: float = Field(0.2, gt=0, lt=1)
    validate_every: int = Field(5, gt=0)
    batch_size: int = Field(32, gt=0, description="Number of samples per gradient update.") # <-- YENİ

class SystemConfig(BaseModel):
    caching_enabled: bool = True
    cache_max_age_hours: int = 24

class StockPredictorConfig(BaseModel):
    """Pydantic model for the stock predictor pipeline configuration."""
    pipeline_name: Literal['stock_predictor']
    data_sourcing: DataSourcingConfig
    feature_engineering: FeatureEngineeringConfig
    model_params: ModelParamsConfig
    training_params: TrainingParamsConfig
    system: SystemConfig

    # Pydantic v2'de ekstra alanlara izin vermemek için:
    class Config:
        extra = 'forbid'