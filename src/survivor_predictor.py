"""
Survivor contestant success predictor using trained models
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from data_processor import SurvivorDataProcessor

class SurvivorPredictor:
    def __init__(self, models_dir=None):
        if models_dir is None:
            models_dir = Path(__file__).parent.parent / 'models'
        
        self.models_dir = Path(models_dir)
        self.processor = SurvivorDataProcessor()
        self.models = {}
        self.preprocessors_loaded = False
        self.load_models()
    
    def load_preprocessors(self):
        """Load preprocessors manually"""
        try:
            preprocessor_file = self.models_dir / 'preprocessors.pkl'
            if preprocessor_file.exists():
                preprocessors = joblib.load(preprocessor_file)
                self.processor.scaler = preprocessors['scaler']
                self.processor.one_hot_encoder = preprocessors['one_hot_encoder']
                self.processor.categorical_features = preprocessors['categorical_features']
                self.processor.numerical_features = preprocessors['numerical_features']
                self.preprocessors_loaded = True
                print("✅ Preprocessors loaded")
                return True
            else:
                print("⚠️ Preprocessors file not found")
                return False
        except Exception as e:
            print(f"Error loading preprocessors: {e}")
            return False
    
    def load_models(self):
        """Load all trained models"""
        try:
            # Load preprocessors
            self.load_preprocessors()
            
            # Load trained models (try tuned first, then regular)
            model_patterns = [
                ('merge_prediction', 'random_forest'),
                ('finale_prediction', 'random_forest'),
                ('winner_prediction', 'random_forest'),
                ('placement_prediction', 'random_forest'),
                ('days_lasted_prediction', 'random_forest')
            ]
            
            for task, model_type in model_patterns:
                # Try tuned model first
                tuned_file = self.models_dir / f"{task}_{model_type}_tuned.pkl"
                regular_file = self.models_dir / f"{task}_{model_type}_model.pkl"
                
                if tuned_file.exists():
                    self.models[task] = joblib.load(tuned_file)
                    print(f"✅ Loaded tuned model for {task}")
                elif regular_file.exists():
                    self.models[task] = joblib.load(regular_file)
                    print(f"✅ Loaded regular model for {task}")
                else:
                    print(f"⚠️ No model found for {task}")
                    
        except Exception as e:
            print(f"Error loading models: {e}")
    
    def predict_contestant_success(self, contestant_data):
        """
        Predict success for a single contestant
        
        Args:
            contestant_data (dict): Dictionary with contestant features
        
        Returns:
            dict: Predictions for all success metrics
        """
        if not self.models:
            print("❌ No models loaded!")
            return None
            
        if not self.preprocessors_loaded:
            print("❌ Preprocessors not loaded!")
            return None
            
        try:
            # Convert to DataFrame
            df = pd.DataFrame([contestant_data])
            
            # Process through the same pipeline as training data
            df = self.processor.create_success_targets(df)
            df = self.processor.create_derived_features(df)
            
            # Preprocess features
            X = self.processor.preprocess_features(df, is_training=False)
            
            # Make predictions
            predictions = {}
            
            for task, model in self.models.items():
                if task in ['merge_prediction', 'finale_prediction', 'winner_prediction']:
                    # Classification - get probability and prediction
                    pred_proba = model.predict_proba(X)[0][1]
                    pred_class = model.predict(X)[0]
                    
                    predictions[task] = {
                        'probability': pred_proba,
                        'prediction': bool(pred_class),
                        'confidence': 'High' if pred_proba > 0.7 or pred_proba < 0.3 else 'Medium'
                    }
                else:
                    # Regression
                    pred_value = model.predict(X)[0]
                    predictions[task] = {
                        'prediction': pred_value,
                        'rounded': round(pred_value)
                    }
            
            return predictions
            
        except Exception as e:
            print(f"Error making predictions: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def create_contestant_report(self, contestant_data):
        """Create a comprehensive success report for a contestant"""
        predictions = self.predict_contestant_success(contestant_data)
        
        if predictions is None:
            return "Unable to generate predictions - check model loading"
        
        report = f"""
🏝️ SURVIVOR SUCCESS PREDICTION REPORT
{'='*50}
Contestant: {contestant_data.get('Contestant_Name', 'Unknown')}
Age: {contestant_data.get('Age', 'Unknown')}
Occupation: {contestant_data.get('Occupation', 'Unknown')}

📊 SUCCESS PREDICTIONS:
{'='*30}
🎯 Make Merge: {predictions['merge_prediction']['probability']:.1%} chance
   Prediction: {'YES' if predictions['merge_prediction']['prediction'] else 'NO'}
   Confidence: {predictions['merge_prediction']['confidence']}

🏆 Make Finale: {predictions['finale_prediction']['probability']:.1%} chance
   Prediction: {'YES' if predictions['finale_prediction']['prediction'] else 'NO'}
   Confidence: {predictions['finale_prediction']['confidence']}

👑 Win Game: {predictions['winner_prediction']['probability']:.1%} chance
   Prediction: {'YES' if predictions['winner_prediction']['prediction'] else 'NO'}
   Confidence: {predictions['winner_prediction']['confidence']}

📅 Days Lasted: ~{predictions['days_lasted_prediction']['rounded']} days
   (Exact prediction: {predictions['days_lasted_prediction']['prediction']:.1f})

🥇 Final Placement: ~{predictions['placement_prediction']['rounded']}
   (Exact prediction: {predictions['placement_prediction']['prediction']:.1f})

💡 SUCCESS TIER ASSESSMENT:
{'='*30}"""

        # Determine success tier
        merge_prob = predictions['merge_prediction']['probability']
        finale_prob = predictions['finale_prediction']['probability']
        winner_prob = predictions['winner_prediction']['probability']
        
        if winner_prob > 0.15:
            tier = "🏆 WINNER POTENTIAL - High chance of victory!"
        elif finale_prob > 0.5:
            tier = "🥈 FINALIST MATERIAL - Likely to make final 3"
        elif merge_prob > 0.7:
            tier = "⭐ STRONG PLAYER - Should make jury"
        elif merge_prob > 0.4:
            tier = "📈 DECENT SHOT - 50/50 merge chances"
        else:
            tier = "⚠️ EARLY BOOT RISK - Needs strong strategy"
        
        report += f"\n{tier}"
        
        return report

def create_sample_contestant():
    """Create a sample contestant for testing"""
    return {
        'Contestant_Name': 'Test Player',
        'Age': 28,
        'Gender': 'M',
        'Occupation': 'Software Engineer',
        'Occupation_Category': 'Business',
        'Home_State': 'CA',
        'Home_Region': 'West',
        'Relationship_Status': 'Single',
        'Athletic_Background': 'College',
        'Physical_Build': 'Medium',
        'Self_Reported_Fitness': 4,
        'Survivor_Knowledge': 'Superfan',
        'Strategic_Archetype': 'Strategic_Player',
        'Pre_Game_Target_Size': 3,
        'Tribal_Challenges_Won': 3,
        'Tribal_Challenges_Total': 5,
        'Individual_Challenges_Won': 2,
        'Individual_Challenges_Total': 8,
        'Advantages_Found': 1,
        'Advantages_Played': 1,
        'Alliance_Count': 2,
        'Votes_Against_Total': 2,
        'Tribals_Attended': 8,
        # Add dummy target values (these will be ignored for prediction)
        'Made_Merge': 'Y',  # Dummy value
        'Made_Finale': 'N',  # Dummy value  
        'Final_Placement': 10,  # Dummy value
        'Days_Lasted': 20,  # Dummy value
        'Elimination_Type': 'Voted_Out',  # Dummy value
        'Jury_Votes_Received': 0  # Dummy value
    }

if __name__ == "__main__":
    print("🏝️ SURVIVOR PREDICTOR TEST")
    print("=" * 30)
    
    predictor = SurvivorPredictor()
    
    if predictor.models and predictor.preprocessors_loaded:
        print("🧪 Testing predictor with sample contestant...")
        
        sample_contestant = create_sample_contestant()
        report = predictor.create_contestant_report(sample_contestant)
        
        print(report)
    else:
        print("❌ Models or preprocessors not loaded properly.")
        print("Available models:", list(predictor.models.keys()))
        print("Preprocessors loaded:", predictor.preprocessors_loaded)
        print("\n💡 Make sure to run model_trainer.py first!")