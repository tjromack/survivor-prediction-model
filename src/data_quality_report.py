"""
Generate comprehensive data quality report
"""
import pandas as pd
import numpy as np
from data_processor import SurvivorDataProcessor

def generate_data_quality_report():
    processor = SurvivorDataProcessor()
    df = processor.load_data()
    X, targets, df_processed = processor.process_full_pipeline(df)
    
    print("=" * 60)
    print("SURVIVOR DATA QUALITY REPORT")
    print("=" * 60)
    
    # Basic dataset info
    print(f"\n📊 DATASET OVERVIEW")
    print(f"Shape: {df.shape}")
    print(f"Seasons: {df['Season'].min()} to {df['Season'].max()}")
    print(f"Contestants per season: {df['Season'].value_counts().iloc[0]}")
    
    # Missing values
    print(f"\n❌ MISSING VALUES")
    missing_summary = df.isnull().sum().sort_values(ascending=False)
    missing_summary = missing_summary[missing_summary > 0]
    if len(missing_summary) == 0:
        print("✅ No missing values found!")
    else:
        print(missing_summary)
    
    # Feature engineering results
    print(f"\n🔧 FEATURE ENGINEERING")
    print(f"Original features: {df.shape[1]}")
    print(f"Processed features: {X.shape[1]}")
    print(f"Features added: {X.shape[1] - df.shape[1] + len(processor.categorical_features)}")
    
    # Target distribution
    print(f"\n🎯 TARGET DISTRIBUTIONS")
    print(f"Made Merge: {targets['made_merge'].sum()}/{len(targets['made_merge'])} ({targets['made_merge'].mean():.1%})")
    print(f"Made Finale: {targets['made_finale'].sum()}/{len(targets['made_finale'])} ({targets['made_finale'].mean():.1%})")
    print(f"Winners: {targets['won_game'].sum()}/{len(targets['won_game'])} ({targets['won_game'].mean():.1%})")
    
    # Success tiers
    print(f"\nSuccess Tier Distribution:")
    tier_counts = targets['success_tier'].value_counts().sort_index()
    for tier, count in tier_counts.items():
        print(f"  {tier}: {count} ({count/len(targets['success_tier']):.1%})")
    
    # Feature types
    print(f"\n🏷️  FEATURE TYPES")
    print(f"Numerical features: {len(processor.numerical_features)}")
    print(f"Categorical features: {len(processor.categorical_features)}")
    
    print(f"\n✅ Data preprocessing complete and ready for modeling!")
    
    return df_processed, X, targets

if __name__ == "__main__":
    df_processed, X, targets = generate_data_quality_report()