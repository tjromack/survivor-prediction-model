"""
Save processed data for modeling
"""
import pandas as pd
import joblib
from data_processor import SurvivorDataProcessor
import sys
sys.path.append('..')
from config.config import *

# Process data
processor = SurvivorDataProcessor()
df = processor.load_data()
X, targets, df_processed = processor.process_full_pipeline(df)

# Save processed data
processed_data = {
    'X': X,
    'targets': targets,
    'df_processed': df_processed
}

joblib.dump(processed_data, PROCESSED_DATA_FILE)
print(f"✅ Processed data saved to {PROCESSED_DATA_FILE}")

# Save preprocessors
processor.save_preprocessors(MODELS_DIR / 'preprocessors.pkl')

print(f"✅ Data preprocessing complete!")
print(f"   Features shape: {X.shape}")
print(f"   Ready for modeling!")