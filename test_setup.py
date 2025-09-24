import pandas as pd
import numpy as np
import sklearn
import xgboost
import matplotlib.pyplot as plt
import seaborn as sns

print("✅ All packages imported successfully!")
print(f"Python packages versions:")
print(f"  pandas: {pd.__version__}")
print(f"  numpy: {np.__version__}")
print(f"  scikit-learn: {sklearn.__version__}")
print(f"  xgboost: {xgboost.__version__}")

# Test data loading
try:
    df = pd.read_csv('data/Survivor_Complete_Dataset_Seasons_4148.csv')
    print(f"✅ Data loaded successfully! Shape: {df.shape}")
except FileNotFoundError:
    print("❌ Data file not found. Please copy your CSV to the data/ directory.")