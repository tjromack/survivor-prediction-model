"""
Model training pipeline for Survivor prediction models
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                           mean_squared_error, r2_score, mean_absolute_error)
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import our data processor
from data_processor import SurvivorDataProcessor

class SurvivorModelTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.feature_importance = {}
        
        # Define model configurations
        self.model_configs = {
            # Classification models
            'merge_prediction': {
                'target': 'made_merge',
                'type': 'classification',
                'models': {
                    'random_forest': RandomForestClassifier(n_estimators=100, random_state=random_state),
                    'xgboost': xgb.XGBClassifier(n_estimators=100, random_state=random_state, eval_metric='logloss'),
                    'logistic': LogisticRegression(random_state=random_state, max_iter=1000)
                }
            },
            'finale_prediction': {
                'target': 'made_finale',
                'type': 'classification',
                'models': {
                    'random_forest': RandomForestClassifier(n_estimators=100, random_state=random_state),
                    'xgboost': xgb.XGBClassifier(n_estimators=100, random_state=random_state, eval_metric='logloss'),
                    'logistic': LogisticRegression(random_state=random_state, max_iter=1000)
                }
            },
            'winner_prediction': {
                'target': 'won_game',
                'type': 'classification',
                'models': {
                    'random_forest': RandomForestClassifier(n_estimators=100, random_state=random_state, class_weight='balanced'),
                    'xgboost': xgb.XGBClassifier(n_estimators=100, random_state=random_state, eval_metric='logloss', scale_pos_weight=10),
                    'logistic': LogisticRegression(random_state=random_state, max_iter=1000, class_weight='balanced')
                }
            },
            # Regression models
            'placement_prediction': {
                'target': 'placement',
                'type': 'regression',
                'models': {
                    'random_forest': RandomForestRegressor(n_estimators=100, random_state=random_state),
                    'xgboost': xgb.XGBRegressor(n_estimators=100, random_state=random_state),
                    'linear': LinearRegression()
                }
            },
            'days_lasted_prediction': {
                'target': 'days_lasted',
                'type': 'regression',
                'models': {
                    'random_forest': RandomForestRegressor(n_estimators=100, random_state=random_state),
                    'xgboost': xgb.XGBRegressor(n_estimators=100, random_state=random_state),
                    'linear': LinearRegression()
                }
            }
        }
    
    def load_processed_data(self):
        """Load the processed data from Phase 2"""
        try:
            # Try to load from the processor first
            processor = SurvivorDataProcessor()
            df = processor.load_data()
            
            if df is not None:
                X, targets, df_processed = processor.process_full_pipeline(df)
                return X, targets
            else:
                # Try to load from saved file
                data_file = Path(__file__).parent.parent / 'data' / 'processed_survivor_data.pkl'
                processed_data = joblib.load(data_file)
                return processed_data['X'], processed_data['targets']
                
        except Exception as e:
            print(f"Error loading processed data: {e}")
            return None, None
    
    def train_classification_model(self, X, y, model, model_name, target_name):
        """Train a classification model with evaluation"""
        print(f"\n🔥 Training {model_name} for {target_name}...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        
        # Store results
        result = {
            'model': model,
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            result['feature_importance'] = importance_df
        
        print(f"   ✅ Accuracy: {accuracy:.3f}")
        print(f"   ✅ CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        return result
    
    def train_regression_model(self, X, y, model, model_name, target_name):
        """Train a regression model with evaluation"""
        print(f"\n📊 Training {model_name} for {target_name}...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Evaluation metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores)
        
        # Store results
        result = {
            'model': model,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'cv_rmse_mean': cv_rmse.mean(),
            'cv_rmse_std': cv_rmse.std(),
            'y_test': y_test,
            'y_pred': y_pred
        }
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            result['feature_importance'] = importance_df
        
        print(f"   ✅ R²: {r2:.3f}")
        print(f"   ✅ RMSE: {rmse:.3f}")
        print(f"   ✅ CV RMSE: {cv_rmse.mean():.3f} (+/- {cv_rmse.std() * 2:.3f})")
        
        return result
    
    def train_all_models(self):
        """Train all configured models"""
        print("🏝️  SURVIVOR MODEL TRAINING PIPELINE")
        print("=" * 50)
        
        # Load data
        X, targets = self.load_processed_data()
        if X is None or targets is None:
            print("❌ Failed to load data!")
            return
        
        print(f"✅ Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Train each model configuration
        for config_name, config in self.model_configs.items():
            print(f"\n🎯 {config_name.upper().replace('_', ' ')}")
            print("-" * 40)
            
            target_name = config['target']
            model_type = config['type']
            
            # Get target variable
            if target_name not in targets:
                print(f"❌ Target '{target_name}' not found!")
                continue
            
            y = targets[target_name]
            
            # Train each model for this target
            config_results = {}
            for model_name, model in config['models'].items():
                try:
                    if model_type == 'classification':
                        result = self.train_classification_model(X, y, model, model_name, target_name)
                    else:
                        result = self.train_regression_model(X, y, model, model_name, target_name)
                    
                    config_results[model_name] = result
                    
                except Exception as e:
                    print(f"❌ Error training {model_name}: {e}")
                    continue
            
            self.results[config_name] = config_results
        
        print(f"\n🎉 Model training complete!")
        return self.results
    
    def get_model_comparison(self):
        """Compare model performances"""
        comparison_data = []
        
        for config_name, config_results in self.results.items():
            for model_name, result in config_results.items():
                if 'accuracy' in result:  # Classification
                    comparison_data.append({
                        'task': config_name,
                        'model': model_name,
                        'metric': 'accuracy',
                        'score': result['accuracy'],
                        'cv_score': result['cv_mean']
                    })
                else:  # Regression
                    comparison_data.append({
                        'task': config_name,
                        'model': model_name,
                        'metric': 'r2',
                        'score': result['r2'],
                        'cv_score': result['cv_rmse_mean']  # Note: this is RMSE, not R2
                    })
        
        return pd.DataFrame(comparison_data)
    
    def plot_model_comparison(self):
        """Visualize model performance comparison"""
        comparison_df = self.get_model_comparison()
        
        if comparison_df.empty:
            print("No results to plot!")
            return
        
        # Create subplots for different tasks
        tasks = comparison_df['task'].unique()
        n_tasks = len(tasks)
        
        fig, axes = plt.subplots(1, n_tasks, figsize=(5*n_tasks, 6))
        if n_tasks == 1:
            axes = [axes]
        
        for i, task in enumerate(tasks):
            task_data = comparison_df[comparison_df['task'] == task]
            
            # Create bar plot
            ax = axes[i]
            bars = ax.bar(task_data['model'], task_data['score'], 
                         alpha=0.7, color=['skyblue', 'lightcoral', 'lightgreen'])
            
            # Add value labels on bars
            for bar, score in zip(bars, task_data['score']):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                       f'{score:.3f}', ha='center', va='bottom')
            
            ax.set_title(f'{task.replace("_", " ").title()}')
            ax.set_ylabel('Score')
            ax.set_ylim(0, 1 if task_data['metric'].iloc[0] == 'accuracy' else None)
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def get_top_features(self, task_name, model_name, top_n=15):
        """Get top features for a specific model"""
        if (task_name in self.results and 
            model_name in self.results[task_name] and 
            'feature_importance' in self.results[task_name][model_name]):
            
            return self.results[task_name][model_name]['feature_importance'].head(top_n)
        else:
            return None
    
    def plot_feature_importance(self, task_name, model_name, top_n=15):
        """Plot feature importance for a specific model"""
        importance_df = self.get_top_features(task_name, model_name, top_n)
        
        if importance_df is None:
            print(f"No feature importance available for {task_name} - {model_name}")
            return
        
        plt.figure(figsize=(10, 8))
        
        # Create horizontal bar plot
        plt.barh(range(len(importance_df)), importance_df['importance'], 
                color='steelblue', alpha=0.7)
        
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Features - {task_name.replace("_", " ").title()} ({model_name})')
        plt.gca().invert_yaxis()
        
        # Add value labels
        for i, (feature, importance) in enumerate(zip(importance_df['feature'], importance_df['importance'])):
            plt.text(importance + 0.001, i, f'{importance:.3f}', 
                    va='center', ha='left')
        
        plt.tight_layout()
        plt.show()
    
    def save_models(self, save_dir):
        """Save all trained models"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        models_saved = 0
        for config_name, config_results in self.results.items():
            for model_name, result in config_results.items():
                model_file = save_dir / f"{config_name}_{model_name}_model.pkl"
                joblib.dump(result['model'], model_file)
                models_saved += 1
        
        # Save results summary
        results_file = save_dir / "training_results.pkl"
        joblib.dump(self.results, results_file)
        
        print(f"✅ Saved {models_saved} models to {save_dir}")
        print(f"✅ Saved training results to {results_file}")

if __name__ == "__main__":
    # Initialize and train models
    trainer = SurvivorModelTrainer()
    results = trainer.train_all_models()
    
    if results:
        # Display comparison
        print("\n📊 MODEL PERFORMANCE COMPARISON")
        print("=" * 50)
        comparison_df = trainer.get_model_comparison()
        print(comparison_df.to_string(index=False))
        
        # Plot comparison
        trainer.plot_model_comparison()
        
        # Show top features for merge prediction
        print("\n🔍 TOP FEATURES FOR MERGE PREDICTION (Random Forest)")
        print("-" * 50)
        merge_features = trainer.get_top_features('merge_prediction', 'random_forest')
        if merge_features is not None:
            print(merge_features.head(10).to_string(index=False))
        
        # Save models
        models_dir = Path(__file__).parent.parent / 'models'
        trainer.save_models(models_dir)
        
        print(f"\n🎉 Training complete! Check the models/ directory for saved models.")