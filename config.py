"""
Configuration file for Survivor Prediction Model
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Data files
RAW_DATA_FILE = DATA_DIR / "Survivor_Complete_Dataset_Seasons_4148.csv"
PROCESSED_DATA_FILE = DATA_DIR / "processed_survivor_data.pkl"

# Model settings
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Target variables for prediction
TARGET_VARIABLES = {
    'placement': 'Final_Placement',
    'merge': 'Made_Merge', 
    'finale': 'Made_Finale',
    'days_lasted': 'Days_Lasted'
}

# Features to exclude from modeling (target leakage or post-game data)
EXCLUDE_FEATURES = [
    'Contestant_Name', 'Season', 'Starting_Tribe', 'Closest_Ally',
    'Final_Placement', 'Elimination_Type', 'Jury_Votes_Received',
    'Made_Merge', 'Made_Finale', 'Days_Lasted', 'Notes',
    'Confessional_Count', 'Screen_Time_Rank'  # Post-game/production data
]

# Model hyperparameters (to be tuned)
MODEL_PARAMS = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': RANDOM_STATE
    },
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': RANDOM_STATE
    }
}