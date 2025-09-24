"""
Data preprocessing pipeline for Survivor prediction model
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import joblib
import sys
import os
from pathlib import Path

# Define configuration directly in this file (no external imports needed)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
RAW_DATA_FILE = DATA_DIR / "Survivor_Complete_Dataset_Seasons_4148.csv"
PROCESSED_DATA_FILE = DATA_DIR / "processed_survivor_data.pkl"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

TARGET_VARIABLES = {
    'placement': 'Final_Placement',
    'merge': 'Made_Merge', 
    'finale': 'Made_Finale',
    'days_lasted': 'Days_Lasted'
}

EXCLUDE_FEATURES = [
    'Contestant_Name', 'Season', 'Starting_Tribe', 'Closest_Ally',
    'Final_Placement', 'Elimination_Type', 'Jury_Votes_Received',
    'Made_Merge', 'Made_Finale', 'Days_Lasted', 'Notes',
    'Confessional_Count', 'Screen_Time_Rank'
]

class SurvivorDataProcessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.categorical_features = []
        self.numerical_features = []
        
    def load_data(self, filepath=None):
        """Load the raw Survivor dataset"""
        if filepath is None:
            filepath = RAW_DATA_FILE
        
        print(f"Loading data from: {filepath}")
        try:
            if not filepath.exists():
                print(f"❌ File not found: {filepath}")
                print("Please ensure your CSV file is named 'Survivor_Complete_Dataset_Seasons_4148.csv'")
                print("and is located in the data/ directory")
                return None
                
            df = pd.read_csv(filepath)
            print(f"✅ Dataset loaded successfully! Shape: {df.shape}")
            return df
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def create_success_targets(self, df):
        """Create multiple success target variables"""
        print("Creating success targets...")
        
        # Binary targets
        df['made_merge_binary'] = (df['Made_Merge'] == 'Y').astype(int)
        df['made_finale_binary'] = (df['Made_Finale'] == 'Y').astype(int)
        df['won_game'] = (df['Final_Placement'] == 1).astype(int)
        
        # Multi-class success tiers
        def success_tier(row):
            if row['Final_Placement'] == 1:
                return 'Winner'
            elif row['Final_Placement'] <= 3:
                return 'Finalist'
            elif row['Made_Merge'] == 'Y':
                return 'Jury'
            elif row['Days_Lasted'] >= 10:
                return 'Mid_Game'
            else:
                return 'Early_Boot'
        
        df['success_tier'] = df.apply(success_tier, axis=1)
        
        # Placement categories (easier for modeling)
        df['placement_category'] = pd.cut(df['Final_Placement'], 
                                        bins=[0, 3, 6, 10, 14, 18], 
                                        labels=['Top_3', 'Top_6', 'Top_10', 'Top_14', 'Bottom_4'])
        
        return df
    
    def create_derived_features(self, df):
        """Engineer new features from existing data"""
        print("Creating derived features...")
        
        # Challenge performance ratios (avoid division by zero)
        df['tribal_win_rate'] = df['Tribal_Challenges_Won'] / (df['Tribal_Challenges_Total'] + 1e-6)
        df['individual_win_rate'] = df['Individual_Challenges_Won'] / (df['Individual_Challenges_Total'] + 1e-6)
        df['total_challenge_wins'] = df['Tribal_Challenges_Won'] + df['Individual_Challenges_Won']
        df['total_challenges'] = df['Tribal_Challenges_Total'] + df['Individual_Challenges_Total']
        df['overall_challenge_rate'] = df['total_challenge_wins'] / (df['total_challenges'] + 1e-6)
        
        # Strategic gameplay metrics
        df['advantages_efficiency'] = df['Advantages_Played'] / (df['Advantages_Found'] + 1e-6)
        df['votes_per_tribal'] = df['Votes_Against_Total'] / (df['Tribals_Attended'] + 1e-6)
        df['survival_rate'] = df['Days_Lasted'] / 26.0  # New era is 26 days
        
        # Age categories
        df['age_group'] = pd.cut(df['Age'], bins=[0, 25, 35, 45, 100], 
                               labels=['Young_Adult', 'Adult', 'Middle_Age', 'Older'])
        
        # Physical metrics (handle missing values)
        physical_build_map = {'Small': 1, 'Medium': 2, 'Large': 3}
        df['physical_build_numeric'] = df['Physical_Build'].map(physical_build_map).fillna(2)
        
        athletic_map = {'None': 0, 'Recreational': 1, 'High_School': 2, 'College': 3, 'Professional': 4}
        df['athletic_background_numeric'] = df['Athletic_Background'].map(athletic_map).fillna(0)
        
        # Strategic archetype groupings
        strategic_groups = {
            'Social_Player': 'Social',
            'Under_Radar': 'Social', 
            'Strategic_Player': 'Strategic',
            'Villain': 'Strategic',
            'Challenge_Beast': 'Physical',
            'Provider': 'Physical',
            'Wild_Card': 'Unpredictable',
            'Hero': 'Social'
        }
        df['strategic_group'] = df['Strategic_Archetype'].map(strategic_groups).fillna('Social')
        
        # Knowledge level impact
        knowledge_map = {'No_Knowledge': 0, 'Casual_Fan': 1, 'Fan': 2, 'Superfan': 3}
        df['knowledge_numeric'] = df['Survivor_Knowledge'].map(knowledge_map).fillna(1)
        
        # Regional advantages (based on historical data)
        competitive_regions = ['CA', 'TX', 'NY', 'FL']  # Use state abbreviations from your data
        df['from_competitive_region'] = df['Home_State'].isin(competitive_regions).astype(int)
        
        return df
    
    def identify_feature_types(self, df):
        """Categorize features for appropriate preprocessing"""
        # Exclude target variables and identifiers
        exclude_cols = set(EXCLUDE_FEATURES + [
            'made_merge_binary', 'made_finale_binary', 'won_game', 
            'success_tier', 'placement_category'
        ])
        
        available_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Categorical features
        categorical = []
        for col in available_cols:
            if (df[col].dtype == 'object' or 
                col.endswith('_group') or 
                col.endswith('_category') or
                col == 'age_group'):
                categorical.append(col)
        
        # Numerical features
        numerical = []
        for col in available_cols:
            if col not in categorical and pd.api.types.is_numeric_dtype(df[col]):
                numerical.append(col)
        
        self.categorical_features = categorical
        self.numerical_features = numerical
        
        print(f"✅ Identified {len(categorical)} categorical features")
        print(f"✅ Identified {len(numerical)} numerical features")
        
        return categorical, numerical
    
    def preprocess_features(self, df, is_training=True):
        """Apply preprocessing to features"""
        df_processed = df.copy()
        
        # Handle missing values
        for col in self.numerical_features:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].fillna(df_processed[col].median())
        
        for col in self.categorical_features:
            if col in df_processed.columns:
                mode_value = df_processed[col].mode()
                fill_value = mode_value[0] if not mode_value.empty else 'Unknown'
                df_processed[col] = df_processed[col].fillna(fill_value)
        
        if is_training:
            # Fit transformers on training data
            
            # One-hot encode categorical features
            if self.categorical_features:
                cat_data = df_processed[self.categorical_features]
                self.one_hot_encoder.fit(cat_data)
                
                # Get feature names for one-hot encoded columns
                feature_names = self.one_hot_encoder.get_feature_names_out(self.categorical_features)
                cat_encoded = self.one_hot_encoder.transform(cat_data)
                cat_df = pd.DataFrame(cat_encoded, columns=feature_names, index=df_processed.index)
            else:
                cat_df = pd.DataFrame(index=df_processed.index)
            
            # Scale numerical features
            if self.numerical_features:
                num_data = df_processed[self.numerical_features]
                num_scaled = self.scaler.fit_transform(num_data)
                num_df = pd.DataFrame(num_scaled, columns=self.numerical_features, index=df_processed.index)
            else:
                num_df = pd.DataFrame(index=df_processed.index)
        else:
            # Transform using fitted transformers
            if self.categorical_features:
                cat_data = df_processed[self.categorical_features]
                cat_encoded = self.one_hot_encoder.transform(cat_data)
                feature_names = self.one_hot_encoder.get_feature_names_out(self.categorical_features)
                cat_df = pd.DataFrame(cat_encoded, columns=feature_names, index=df_processed.index)
            else:
                cat_df = pd.DataFrame(index=df_processed.index)
            
            if self.numerical_features:
                num_data = df_processed[self.numerical_features]
                num_scaled = self.scaler.transform(num_data)
                num_df = pd.DataFrame(num_scaled, columns=self.numerical_features, index=df_processed.index)
            else:
                num_df = pd.DataFrame(index=df_processed.index)
        
        # Combine processed features
        X = pd.concat([num_df, cat_df], axis=1)
        
        return X
    
    def prepare_targets(self, df):
        """Prepare target variables"""
        targets = {}
        targets['placement'] = df['Final_Placement']
        targets['days_lasted'] = df['Days_Lasted']
        targets['made_merge'] = df['made_merge_binary']
        targets['made_finale'] = df['made_finale_binary']
        targets['won_game'] = df['won_game']
        targets['success_tier'] = df['success_tier']
        targets['placement_category'] = df['placement_category']
        
        return targets
    
    def process_full_pipeline(self, df):
        """Run the complete preprocessing pipeline"""
        print("\n🏝️  Starting data preprocessing pipeline...")
        print("-" * 50)
        
        if df is None:
            print("❌ No data to process!")
            return None, None, None
        
        try:
            # Create success targets
            df = self.create_success_targets(df)
            print("✅ Success targets created")
            
            # Create derived features
            df = self.create_derived_features(df)
            print("✅ Derived features created")
            
            # Identify feature types
            self.identify_feature_types(df)
            print("✅ Feature types identified")
            
            # Preprocess features
            X = self.preprocess_features(df, is_training=True)
            print("✅ Features preprocessed")
            
            # Prepare targets
            targets = self.prepare_targets(df)
            print("✅ Targets prepared")
            
            print(f"\n📊 Final feature matrix shape: {X.shape}")
            print(f"📋 Feature columns: {len(X.columns)} total")
            
            return X, targets, df
            
        except Exception as e:
            print(f"❌ Error in preprocessing pipeline: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None
    
    def save_preprocessors(self, filepath):
        """Save fitted preprocessors"""
        # Ensure directory exists
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        preprocessors = {
            'scaler': self.scaler,
            'one_hot_encoder': self.one_hot_encoder,
            'categorical_features': self.categorical_features,
            'numerical_features': self.numerical_features
        }
        joblib.dump(preprocessors, filepath)
        print(f"✅ Preprocessors saved to {filepath}")

if __name__ == "__main__":
    print("🏝️  SURVIVOR PREDICTION MODEL - DATA PROCESSOR")
    print("=" * 55)
    
    # Test the processor
    processor = SurvivorDataProcessor()
    df = processor.load_data()
    
    if df is not None:
        X, targets, df_processed = processor.process_full_pipeline(df)
        
        if X is not None and targets is not None:
            print("\n🎉 SUCCESS! Data preprocessing completed.")
            print("=" * 55)
            print(f"📊 PROCESSING RESULTS:")
            print(f"   Original data shape: {df.shape}")
            print(f"   Processed features shape: {X.shape}")
            print(f"   Feature count: {X.shape[1]}")
            
            print(f"\n🎯 TARGET VARIABLES:")
            for name, target in targets.items():
                if target.dtype == 'object' or str(target.dtype) == 'category':
                    unique_vals = target.value_counts()
                    print(f"   {name}: {dict(unique_vals)}")
                else:
                    print(f"   {name}: mean={target.mean():.3f}, std={target.std():.3f}")
            
            # Save processed data for later use
            try:
                processed_data = {
                    'X': X,
                    'targets': targets,
                    'df_processed': df_processed
                }
                
                # Ensure data directory exists
                DATA_DIR.mkdir(exist_ok=True)
                joblib.dump(processed_data, PROCESSED_DATA_FILE)
                print(f"\n💾 Processed data saved to: {PROCESSED_DATA_FILE}")
                
                # Save preprocessors
                MODELS_DIR.mkdir(exist_ok=True)
                processor.save_preprocessors(MODELS_DIR / 'preprocessors.pkl')
                
            except Exception as e:
                print(f"⚠️  Warning: Could not save processed data: {e}")
            
        else:
            print("❌ Processing failed!")
    else:
        print("❌ Failed to load data! Please check:")
        print("   1. CSV file is in data/ directory")
        print("   2. Filename is exactly 'Survivor_Complete_Dataset_Seasons_4148.csv'")
        print("   3. Virtual environment is activated")