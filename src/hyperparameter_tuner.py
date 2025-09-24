"""
Hyperparameter tuning for Survivor prediction models
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from model_trainer import SurvivorModelTrainer

class SurvivorHyperparameterTuner:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.best_models = {}
        
        # Define hyperparameter grids
        self.param_grids = {
            'random_forest_classifier': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            },
            'xgb_classifier': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            },
            'logistic_regression': {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'random_forest_regressor': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            },
            'xgb_regressor': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
        }
    
    def tune_model(self, model, param_grid, X, y, scoring, model_name, cv=5):
        """Tune hyperparameters for a single model"""
        print(f"🔧 Tuning {model_name}...")
        
        # Use RandomizedSearchCV for efficiency
        search = RandomizedSearchCV(
            model, param_grid, 
            n_iter=50,  # Number of parameter combinations to try
            cv=cv, 
            scoring=scoring, 
            random_state=self.random_state,
            n_jobs=-1,
            verbose=0
        )
        
        search.fit(X, y)
        
        print(f"   ✅ Best score: {search.best_score_:.3f}")
        print(f"   ✅ Best params: {search.best_params_}")
        
        return search.best_estimator_, search.best_score_, search.best_params_
    
    def tune_all_models(self):
        """Tune hyperparameters for all key models"""
        print("🏝️  SURVIVOR MODEL HYPERPARAMETER TUNING")
        print("=" * 55)
        
        # Load data
        trainer = SurvivorModelTrainer()
        X, targets = trainer.load_processed_data()
        
        if X is None or targets is None:
            print("❌ Failed to load data!")
            return
        
        print(f"✅ Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Focus on key prediction tasks
        key_tasks = {
            'merge_prediction': {
                'target': targets['made_merge'],
                'type': 'classification',
                'scoring': 'accuracy'
            },
            'finale_prediction': {
                'target': targets['made_finale'], 
                'type': 'classification',
                'scoring': 'accuracy'
            },
            'placement_prediction': {
                'target': targets['placement'],
                'type': 'regression',
                'scoring': 'neg_mean_squared_error'
            }
        }
        
        for task_name, task_info in key_tasks.items():
            print(f"\n🎯 {task_name.upper().replace('_', ' ')}")
            print("-" * 40)
            
            y = task_info['target']
            task_type = task_info['type']
            scoring = task_info['scoring']
            
            task_results = {}
            
            if task_type == 'classification':
                # Tune classification models
                models_to_tune = {
                    'random_forest': RandomForestClassifier(random_state=self.random_state),
                    'xgboost': xgb.XGBClassifier(random_state=self.random_state, eval_metric='logloss'),
                    'logistic': LogisticRegression(random_state=self.random_state, max_iter=1000)
                }
                
                param_grids = {
                    'random_forest': self.param_grids['random_forest_classifier'],
                    'xgboost': self.param_grids['xgb_classifier'], 
                    'logistic': self.param_grids['logistic_regression']
                }
                
            else:  # regression
                models_to_tune = {
                    'random_forest': RandomForestRegressor(random_state=self.random_state),
                    'xgboost': xgb.XGBRegressor(random_state=self.random_state)
                }
                
                param_grids = {
                    'random_forest': self.param_grids['random_forest_regressor'],
                    'xgboost': self.param_grids['xgb_regressor']
                }
            
            # Tune each model
            for model_name, model in models_to_tune.items():
                try:
                    best_model, best_score, best_params = self.tune_model(
                        model, param_grids[model_name], X, y, scoring, model_name
                    )
                    
                    task_results[model_name] = {
                        'model': best_model,
                        'score': best_score,
                        'params': best_params
                    }
                    
                except Exception as e:
                    print(f"❌ Error tuning {model_name}: {e}")
                    continue
            
            self.best_models[task_name] = task_results
        
        print(f"\n🎉 Hyperparameter tuning complete!")
        return self.best_models
    
    def save_tuned_models(self, save_dir):
        """Save the best tuned models"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        models_saved = 0
        for task_name, task_results in self.best_models.items():
            for model_name, result in task_results.items():
                model_file = save_dir / f"{task_name}_{model_name}_tuned.pkl"
                joblib.dump(result['model'], model_file)
                models_saved += 1
        
        # Save tuning results
        tuning_results_file = save_dir / "tuning_results.pkl" 
        joblib.dump(self.best_models, tuning_results_file)
        
        print(f"✅ Saved {models_saved} tuned models to {save_dir}")
        print(f"✅ Saved tuning results to {tuning_results_file}")

if __name__ == "__main__":
    tuner = SurvivorHyperparameterTuner()
    best_models = tuner.tune_all_models()
    
    if best_models:
        print("\n🏆 BEST MODEL RESULTS")
        print("=" * 50)
        
        for task_name, task_results in best_models.items():
            print(f"\n{task_name.replace('_', ' ').title()}:")
            for model_name, result in task_results.items():
                print(f"  {model_name}: {result['score']:.3f}")
        
        # Save tuned models
        models_dir = Path(__file__).parent.parent / 'models'
        tuner.save_tuned_models(models_dir)