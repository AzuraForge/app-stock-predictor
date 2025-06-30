# app-stock-predictor/src/azuraforge_stockapp/caching.py

import logging
import os
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional
import pandas as pd

def get_cache_filepath(cache_dir: str, context: str, params: Dict[str, Any]) -> str:
    """
    Verilen parametrelere göre deterministik bir önbellek dosya yolu oluşturur.
    """
    # Parametreleri tutarlı bir şekilde sıralayıp string'e çevir
    param_str = str(sorted(params.items()))
    param_hash = hashlib.md5(param_str.encode()).hexdigest()
    filename = f"{context}_{param_hash}.parquet"
    
    # Bağlama özel bir alt klasör oluştur
    full_cache_dir = os.path.join(cache_dir, context)
    os.makedirs(full_cache_dir, exist_ok=True)
    
    return os.path.join(full_cache_dir, filename)

def load_from_cache(filepath: str, max_age_hours: int) -> Optional[pd.DataFrame]:
    """Veriyi önbellekten yükler, süresi dolmuşsa veya yoksa None döner."""
    if not os.path.exists(filepath):
        return None
        
    try:
        mod_time = datetime.fromtimestamp(os.path.getmtime(filepath), tz=timezone.utc)
        if (datetime.now(timezone.utc) - mod_time) < timedelta(hours=max_age_hours):
            logging.info(f"Geçerli önbellek bulundu, buradan okunuyor: {filepath}")
            return pd.read_parquet(filepath)
        else:
            logging.info(f"Önbellek süresi dolmuş: {filepath}")
            return None
    except Exception as e:
        logging.error(f"Önbellekten okuma hatası {filepath}: {e}")
        return None

def save_to_cache(df: pd.DataFrame, filepath: str) -> None:
    """DataFrame'i önbelleğe kaydeder."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_parquet(filepath)
        logging.info(f"Veri önbelleğe kaydedildi: {filepath}")
    except Exception as e:
        logging.error(f"Önbelleğe yazma hatası {filepath}: {e}")