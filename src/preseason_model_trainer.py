"""
Pre-Season Survivor Prediction Model
Trained only on demographics and pre-game characteristics
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, r2_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from data_processor import SurvivorDataProcessor

class PreSeasonModelTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.label_encoders = {}
        
        # Define PRE-SEASON ONLY features
        self.preseason_features = [
            'Age',
            'Gender',
            'Occupation_Category', 
            'Home_Region',
            'Relationship_Status',
            'Athletic_Background',
            'Physical_Build',
            'Self_Reported_Fitness',
            'Survivor_Knowledge',
            'Strategic_Archetype',
            'Pre_Game_Target_Size',
            # Derived demographic features
            'age_group',
            'physical_build_numeric',
            'athletic_background_numeric',
            'strategic_group',
            'knowledge_numeric',
            'from_competitive_region'
        ]
    
    def load_and_prepare_data(self):
        """Load data and prepare pre-season only features"""
        processor = SurvivorDataProcessor()
        df = processor.load_data()
        
        if df is None:
            return None, None
        
        # Create derived features
        df = processor.create_derived_features(df)
        
        # Create success targets  
        df = processor.create_success_targets(df)
        
        # Select only pre-season features
        available_features = [f for f in self.preseason_features if f in df.columns]
        X_raw = df[available_features].copy()
        
        print(f"Using {len(available_features)} pre-season features:")
        for feature in available_features:
            print(f"  - {feature}")
        
        # Encode categorical features manually
        X = X_raw.copy()
        categorical_features = []
        
        for col in X.columns:
            if X[col].dtype == 'object' or col in ['age_group', 'strategic_group']:
                categorical_features.append(col)
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
                else:
                    X[col] = self.label_encoders[col].transform(X[col].astype(str))
        
        print(f"Encoded {len(categorical_features)} categorical features")
        
        # Prepare targets
        targets = {
            'made_merge': df['made_merge_binary'],
            'made_finale': df['made_finale_binary'],
            'won_game': df['won_game'],
            'placement': df['Final_Placement'],
            'days_lasted': df['Days_Lasted']
        }
        
        return X, targets
    
    def train_preseason_models(self):
        """Train models using only pre-season data"""
        print("🏝️ TRAINING PRE-SEASON SURVIVOR MODELS")
        print("=" * 50)
        
        X, targets = self.load_and_prepare_data()
        if X is None:
            return None
        
        print(f"Training on {X.shape[0]} contestants with {X.shape[1]} pre-season features")
        
        # Model configurations for pre-season prediction
        model_configs = {
            'merge_prediction': {
                'target': targets['made_merge'],
                'type': 'classification',
                'models': {
                    'random_forest': RandomForestClassifier(
                        n_estimators=200, max_depth=8, min_samples_split=10,
                        random_state=self.random_state, class_weight='balanced'
                    ),
                    'logistic': LogisticRegression(
                        random_state=self.random_state, max_iter=1000, 
                        class_weight='balanced', C=0.1
                    )
                }
            },
            'finale_prediction': {
                'target': targets['made_finale'], 
                'type': 'classification',
                'models': {
                    'random_forest': RandomForestClassifier(
                        n_estimators=200, max_depth=6, min_samples_split=15,
                        random_state=self.random_state, class_weight='balanced'
                    ),
                    'logistic': LogisticRegression(
                        random_state=self.random_state, max_iter=1000,
                        class_weight='balanced', C=0.01
                    )
                }
            },
            'winner_prediction': {
                'target': targets['won_game'],
                'type': 'classification', 
                'models': {
                    'random_forest': RandomForestClassifier(
                        n_estimators=300, max_depth=4, min_samples_split=20,
                        random_state=self.random_state, class_weight='balanced_subsample'
                    ),
                    'logistic': LogisticRegression(
                        random_state=self.random_state, max_iter=1000,
                        class_weight='balanced', C=0.001
                    )
                }
            }
        }
        
        # Train each model
        for task_name, config in model_configs.items():
            print(f"\n🎯 {task_name.upper().replace('_', ' ')}")
            print("-" * 40)
            
            y = config['target']
            task_results = {}
            
            for model_name, model in config['models'].items():
                print(f"Training {model_name}...")
                
                # Train-test split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.25, random_state=self.random_state, 
                    stratify=y if config['type'] == 'classification' else None
                )
                
                # Fit model
                model.fit(X_train, y_train)
                
                # Evaluate
                if config['type'] == 'classification':
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
                    
                    print(f"  ✅ {model_name}: {accuracy:.3f} accuracy (CV: {cv_scores.mean():.3f})")
                    
                    # Feature importance
                    if hasattr(model, 'feature_importances_'):
                        importance_df = pd.DataFrame({
                            'feature': X.columns,
                            'importance': model.feature_importances_
                        }).sort_values('importance', ascending=False)
                        
                        print(f"  Top 5 features: {list(importance_df.head()['feature'])}")
                    
                    task_results[model_name] = {
                        'model': model,
                        'accuracy': accuracy,
                        'cv_mean': cv_scores.mean(),
                        'feature_importance': importance_df if hasattr(model, 'feature_importances_') else None
                    }
            
            self.results[task_name] = task_results
        
        return self.results
    
    def predict_preseason_contestant(self, contestant_data):
        """Make pre-season prediction for a contestant"""
        
        # Prepare contestant data
        contestant_df = pd.DataFrame([contestant_data])
        
        # Create derived features (same as training)
        processor = SurvivorDataProcessor()
        contestant_df = processor.create_derived_features(contestant_df)
        
        # Select pre-season features
        available_features = [f for f in self.preseason_features if f in contestant_df.columns]
        X_contestant = contestant_df[available_features].copy()
        
        # Encode categorical features
        for col in X_contestant.columns:
            if X_contestant[col].dtype == 'object' or col in ['age_group', 'strategic_group']:
                if col in self.label_encoders:
                    X_contestant[col] = self.label_encoders[col].transform(X_contestant[col].astype(str))
                else:
                    # Handle unseen categories
                    X_contestant[col] = 0
        
        # Make predictions
        predictions = {}
        
        for task_name, task_results in self.results.items():
            # Use best performing model (random forest typically)
            best_model = task_results['random_forest']['model']
            
            pred_proba = best_model.predict_proba(X_contestant)[0]
            pred_class = best_model.predict(X_contestant)[0]
            
            predictions[task_name] = {
                'probability': pred_proba[1] if len(pred_proba) > 1 else pred_proba[0],
                'prediction': bool(pred_class)
            }
        
        return predictions
    
    def save_preseason_models(self, save_dir):
        """Save pre-season models"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save models
        for task_name, task_results in self.results.items():
            for model_name, result in task_results.items():
                model_file = save_dir / f"preseason_{task_name}_{model_name}.pkl"
                joblib.dump(result['model'], model_file)
        
        # Save label encoders
        encoders_file = save_dir / "preseason_label_encoders.pkl"
        joblib.dump(self.label_encoders, encoders_file)
        
        # Save feature list
        features_file = save_dir / "preseason_features.pkl"  
        joblib.dump(self.preseason_features, features_file)
        
        print(f"✅ Pre-season models saved to {save_dir}")

if __name__ == "__main__":
    trainer = PreSeasonModelTrainer()
    results = trainer.train_preseason_models()
    
    if results:
        print("\n📊 PRE-SEASON MODEL RESULTS")
        print("=" * 50)
        
        for task_name, task_results in results.items():
            print(f"\n{task_name.replace('_', ' ').title()}:")
            for model_name, result in task_results.items():
                if 'accuracy' in result:
                    print(f"  {model_name}: {result['accuracy']:.3f} accuracy")
        
        # Save models
        models_dir = Path(__file__).parent.parent / 'models'
        trainer.save_preseason_models(models_dir)
        
        print(f"\n🎉 Pre-season models trained and saved!")
        print("These models focus on demographics and pre-game characteristics only.")