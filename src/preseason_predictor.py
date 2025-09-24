"""
Pre-Season Survivor Contestant Predictor
Uses only demographic and pre-game characteristics
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from data_processor import SurvivorDataProcessor

class PreSeasonPredictor:
    def __init__(self, models_dir=None):
        if models_dir is None:
            models_dir = Path(__file__).parent.parent / 'models'
        
        self.models_dir = Path(models_dir)
        self.models = {}
        self.label_encoders = {}
        self.preseason_features = []
        self.processor = SurvivorDataProcessor()
        
        self.load_preseason_models()
    
    def load_preseason_models(self):
        """Load pre-season specific models"""
        try:
            # Load label encoders
            encoders_file = self.models_dir / 'preseason_label_encoders.pkl'
            if encoders_file.exists():
                self.label_encoders = joblib.load(encoders_file)
                print("✅ Pre-season label encoders loaded")
            
            # Load feature list
            features_file = self.models_dir / 'preseason_features.pkl'
            if features_file.exists():
                self.preseason_features = joblib.load(features_file)
                print("✅ Pre-season features list loaded")
            
            # Load models
            model_patterns = [
                ('merge_prediction', 'random_forest'),
                ('finale_prediction', 'random_forest'), 
                ('winner_prediction', 'random_forest')
            ]
            
            for task, model_type in model_patterns:
                model_file = self.models_dir / f"preseason_{task}_{model_type}.pkl"
                if model_file.exists():
                    self.models[task] = joblib.load(model_file)
                    print(f"✅ Loaded pre-season model for {task}")
                else:
                    print(f"⚠️ No pre-season model found for {task}")
        
        except Exception as e:
            print(f"Error loading pre-season models: {e}")
    
    def predict_contestant_preseason(self, contestant_data):
        """Predict success using only pre-season characteristics"""
        
        if not self.models:
            print("❌ No pre-season models loaded")
            return None
        
        try:
            # Convert to DataFrame
            contestant_df = pd.DataFrame([contestant_data])
            
            # Add dummy gameplay data to prevent errors in derived features
            gameplay_defaults = {
                'Tribal_Challenges_Won': 0,
                'Tribal_Challenges_Total': 0,
                'Individual_Challenges_Won': 0,
                'Individual_Challenges_Total': 0,
                'Advantages_Found': 0,
                'Advantages_Played': 0,
                'Alliance_Count': 0,
                'Votes_Against_Total': 0,
                'Tribals_Attended': 0,
                'Days_Lasted': 26,
                'Final_Placement': 10,
                'Made_Merge': 'Y',
                'Made_Finale': 'N',
                'Elimination_Type': 'Voted_Out',
                'Jury_Votes_Received': 0
            }
            
            # Add missing columns with defaults
            for col, default_val in gameplay_defaults.items():
                if col not in contestant_df.columns:
                    contestant_df[col] = default_val
            
            # Create derived features (this needs the gameplay columns)
            contestant_df = self.processor.create_derived_features(contestant_df)
            
            # Now select ONLY the pre-season features we actually trained on
            preseason_only = [
                'Age', 'Gender', 'Occupation_Category', 'Home_Region',
                'Relationship_Status', 'Athletic_Background', 'Physical_Build',
                'Self_Reported_Fitness', 'Survivor_Knowledge', 'Strategic_Archetype',
                'Pre_Game_Target_Size', 'age_group', 'physical_build_numeric',
                'athletic_background_numeric', 'strategic_group',
                'knowledge_numeric', 'from_competitive_region'
            ]
            
            available_features = [f for f in preseason_only if f in contestant_df.columns]
            X_contestant = contestant_df[available_features].copy()
            
            print(f"Using {len(available_features)} pre-season features for prediction")
            
            # Handle missing pre-season features
            for feature in preseason_only:
                if feature not in X_contestant.columns:
                    if 'numeric' in feature or feature in ['Age', 'Self_Reported_Fitness', 'Pre_Game_Target_Size', 'from_competitive_region']:
                        X_contestant[feature] = 0 if 'from_competitive' in feature else 3
                    elif feature == 'age_group':
                        X_contestant[feature] = 'Adult'
                    elif feature == 'strategic_group':
                        X_contestant[feature] = 'Social'
                    else:
                        X_contestant[feature] = 'Unknown'
            
            # Encode categorical features using the same encoding as training
            for col in X_contestant.columns:
                if X_contestant[col].dtype == 'object' or col in ['age_group', 'strategic_group']:
                    if col in self.label_encoders:
                        try:
                            # Handle the transform carefully
                            unique_vals = X_contestant[col].unique()
                            encoded_vals = []
                            for val in X_contestant[col]:
                                try:
                                    encoded_vals.append(self.label_encoders[col].transform([str(val)])[0])
                                except ValueError:
                                    # Unseen category - use most common encoding (usually 0)
                                    encoded_vals.append(0)
                            X_contestant[col] = encoded_vals
                        except Exception as e:
                            print(f"Encoding error for {col}: {e}")
                            X_contestant[col] = 0
                    else:
                        # No encoder available - convert to numeric
                        X_contestant[col] = 0
            
            # Ensure all features are numeric
            for col in X_contestant.columns:
                X_contestant[col] = pd.to_numeric(X_contestant[col], errors='coerce').fillna(0)
            
            # Make predictions
            predictions = {}
            
            for task_name, model in self.models.items():
                try:
                    # Get probability prediction
                    pred_proba = model.predict_proba(X_contestant)[0]
                    pred_class = model.predict(X_contestant)[0]
                    
                    # Extract positive class probability
                    positive_prob = pred_proba[1] if len(pred_proba) > 1 else pred_proba[0]
                    
                    predictions[task_name] = {
                        'probability': positive_prob,
                        'prediction': bool(pred_class),
                        'confidence': 'High' if positive_prob > 0.7 or positive_prob < 0.3 else 'Medium'
                    }
                    
                    print(f"{task_name}: {positive_prob:.3f} probability")
                    
                except Exception as e:
                    print(f"Error predicting {task_name}: {e}")
                    predictions[task_name] = {
                        'probability': 0.5,
                        'prediction': False,
                        'confidence': 'Low'
                    }
            
            return predictions
            
        except Exception as e:
            print(f"Error in pre-season prediction: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def create_preseason_report(self, contestant_data):
        """Create a pre-season focused prediction report"""
        predictions = self.predict_contestant_preseason(contestant_data)
        
        if predictions is None:
            return "Unable to generate pre-season predictions"
        
        name = contestant_data.get('Contestant_Name', 'Unknown')
        age = contestant_data.get('Age', 'Unknown')
        occupation = contestant_data.get('Occupation', 'Unknown')
        archetype = contestant_data.get('Strategic_Archetype', 'Unknown')
        
        report = f"""
🏝️ PRE-SEASON SURVIVOR PREDICTION
{'='*50}
Contestant: {name}
Age: {age}
Occupation: {occupation}  
Strategic Archetype: {archetype}

📊 PRE-SEASON PREDICTIONS (Based on Demographics & Player Type):
{'='*50}
🎯 Make Merge: {predictions.get('merge_prediction', {}).get('probability', 0):.1%} chance
   Prediction: {'YES' if predictions.get('merge_prediction', {}).get('prediction', False) else 'NO'}
   Confidence: {predictions.get('merge_prediction', {}).get('confidence', 'Unknown')}

🏆 Make Finale: {predictions.get('finale_prediction', {}).get('probability', 0):.1%} chance  
   Prediction: {'YES' if predictions.get('finale_prediction', {}).get('prediction', False) else 'NO'}
   Confidence: {predictions.get('finale_prediction', {}).get('confidence', 'Unknown')}

👑 Win Game: {predictions.get('winner_prediction', {}).get('probability', 0):.1%} chance
   Prediction: {'YES' if predictions.get('winner_prediction', {}).get('prediction', False) else 'NO'}
   Confidence: {predictions.get('winner_prediction', {}).get('confidence', 'Unknown')}

💡 PRE-SEASON ASSESSMENT:
{'='*50}
This prediction is based solely on:
• Demographics (age, location, background)
• Physical characteristics (build, fitness, athletic history)
• Strategic profile (archetype, Survivor knowledge, target size)
• Occupation and life experience

⚠️ Note: Pre-season predictions have higher uncertainty than in-game predictions.
Real success depends heavily on social dynamics, challenge performance, and luck.
        """
        
        return report

def create_test_contestant():
    """Create a test contestant for pre-season prediction"""
    return {
        'Contestant_Name': 'Alex Moore',
        'Age': 27,
        'Gender': 'M',
        'Occupation': 'Political comms director',
        'Occupation_Category': 'Government',
        'Home_State': 'IL',
        'Home_Region': 'Midwest', 
        'Relationship_Status': 'Single',
        'Athletic_Background': 'Recreational',
        'Physical_Build': 'Medium',
        'Self_Reported_Fitness': 4,
        'Survivor_Knowledge': 'Fan',
        'Strategic_Archetype': 'Strategic_Player',
        'Pre_Game_Target_Size': 4
    }

if __name__ == "__main__":
    print("🏝️ TESTING PRE-SEASON PREDICTOR")
    print("=" * 40)
    
    predictor = PreSeasonPredictor()
    
    if predictor.models:
        test_contestant = create_test_contestant()
        report = predictor.create_preseason_report(test_contestant)
        print(report)
    else:
        print("❌ Pre-season models not loaded. Run preseason_model_trainer.py first!")