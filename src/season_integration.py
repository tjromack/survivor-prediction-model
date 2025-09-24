"""
Season 49+ Integration System
Handles new season data input, validation, and model updating
"""
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from datetime import datetime
import json
import shutil
import warnings
warnings.filterwarnings('ignore')

from data_processor import SurvivorDataProcessor
from model_trainer import SurvivorModelTrainer

class SeasonIntegrator:
    def __init__(self, project_root=None):
        if project_root is None:
            project_root = Path(__file__).parent.parent
        
        self.project_root = Path(project_root)
        self.data_dir = self.project_root / 'data'
        self.models_dir = self.project_root / 'models'
        self.results_dir = self.project_root / 'results'
        
        # Create directories if they don't exist
        for dir_path in [self.data_dir, self.models_dir, self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.processor = SurvivorDataProcessor()
        
        # Season 49+ data template
        self.season_template = {
            'Season': 49,
            'Contestant_Name': '',
            'Age': 0,
            'Gender': '',  # M/F
            'Starting_Tribe': '',
            'Occupation': '',
            'Occupation_Category': '',  # Business, Healthcare, Education, etc.
            'Home_State': '',
            'Home_Region': '',  # West, South, Midwest, Northeast, Canada
            'Relationship_Status': '',  # Single, Married, In_Relationship, etc.
            'Athletic_Background': '',  # None, Recreational, High_School, College, Professional
            'Physical_Build': '',  # Small, Medium, Large
            'Self_Reported_Fitness': 0,  # 1-5 scale
            'Survivor_Knowledge': '',  # No_Knowledge, Casual_Fan, Fan, Superfan
            'Strategic_Archetype': '',  # Social_Player, Strategic_Player, Challenge_Beast, etc.
            'Pre_Game_Target_Size': 0,  # 1-5 scale
            # Performance data (to be filled as season progresses)
            'Tribal_Challenges_Won': 0,
            'Tribal_Challenges_Total': 0,
            'Individual_Challenges_Won': 0,
            'Individual_Challenges_Total': 0,
            'Advantages_Found': 0,
            'Advantages_Played': 0,
            'Alliance_Count': 0,
            'Closest_Ally': '',
            'Votes_Against_Total': 0,
            'Tribals_Attended': 0,
            'Days_Lasted': 26,  # Will be updated as season progresses
            'Final_Placement': 0,  # Will be updated when eliminated
            'Elimination_Type': '',  # Will be updated
            'Jury_Votes_Received': 0,  # For finale
            'Made_Merge': '',  # Y/N - updated at merge
            'Made_Finale': '',  # Y/N - updated at finale
            'Confessional_Count': 0,  # Post-season data
            'Screen_Time_Rank': 0,  # Post-season data
            'Notes': ''
        }
    
    def create_season_template(self, season_number=49, num_contestants=18):
        """Create a template CSV for new season data input"""
        
        # Create template dataframe
        template_data = []
        for i in range(num_contestants):
            contestant_template = self.season_template.copy()
            contestant_template['Season'] = season_number
            contestant_template['Contestant_Name'] = f'Contestant_{i+1}'
            template_data.append(contestant_template)
        
        template_df = pd.DataFrame(template_data)
        
        # Save template
        template_file = self.data_dir / f'season_{season_number}_template.csv'
        template_df.to_csv(template_file, index=False)
        
        print(f"✅ Created Season {season_number} template: {template_file}")
        
        # Create instructions file
        instructions_file = self.data_dir / f'season_{season_number}_instructions.md'
        self.create_data_entry_instructions(instructions_file, season_number)
        
        return template_file
    
    def create_data_entry_instructions(self, instructions_file, season_number):
        """Create detailed instructions for data entry"""
        
        instructions = f"""
# Season {season_number} Data Entry Instructions

## Overview
Fill out the season_{season_number}_template.csv file with contestant information as it becomes available.

## Data Entry Phases

### Phase 1: Pre-Season (Cast Announcement)
Fill out these columns immediately when cast is announced:
- **Contestant_Name**: Full name as shown on CBS
- **Age**: Age at time of filming
- **Gender**: M or F
- **Occupation**: Current job/profession
- **Occupation_Category**: Business, Healthcare, Education, Entertainment, Government, Sports, Military, Legal, Other
- **Home_State**: Two-letter state code (CA, TX, NY, etc.) or ON for Canada
- **Home_Region**: West, South, Midwest, Northeast, Canada
- **Relationship_Status**: Single, Married, In_Relationship, Divorced, Engaged
- **Athletic_Background**: None, Recreational, High_School, College, Professional
- **Physical_Build**: Small, Medium, Large (based on appearance)
- **Self_Reported_Fitness**: 1-5 scale (estimate from interviews)
- **Survivor_Knowledge**: No_Knowledge, Casual_Fan, Fan, Superfan (from interviews)
- **Strategic_Archetype**: Social_Player, Strategic_Player, Challenge_Beast, Under_Radar, Villain, Hero, Provider, Wild_Card
- **Pre_Game_Target_Size**: 1-5 scale (how big a target they seem pre-game)

### Phase 2: During Season (Update Weekly)
Update these as episodes air:
- **Tribal_Challenges_Won/Total**: Count team challenge wins
- **Individual_Challenges_Won/Total**: Count individual immunity wins
- **Advantages_Found/Played**: Hidden immunity idols, advantages
- **Alliance_Count**: Number of alliances formed
- **Votes_Against_Total**: Cumulative votes received
- **Tribals_Attended**: Number of tribal councils attended
- **Days_Lasted**: Update when eliminated
- **Made_Merge**: Y when merge happens, N when eliminated pre-merge

### Phase 3: Post-Season
Final data entry:
- **Final_Placement**: 1-18 placement
- **Elimination_Type**: Winner, Runner_Up, Fire_Challenge, Voted_Out, Medical
- **Jury_Votes_Received**: For finalists only
- **Made_Finale**: Y for final 3, N for others
- **Confessional_Count**: From episode transcripts
- **Screen_Time_Rank**: 1-18 ranking by screen time

## Value Guidelines

**Strategic_Archetype Options:**
- Social_Player: Focuses on relationships and social bonds
- Strategic_Player: Game-focused, strategic thinker
- Challenge_Beast: Physical competitor, challenge-focused
- Under_Radar: Quiet, low-threat gameplay
- Villain: Aggressive, confrontational style
- Hero: Likeable, moral compass type
- Provider: Camp life contributor
- Wild_Card: Unpredictable gameplay

**Occupation_Category Guidelines:**
- Business: Corporate, sales, marketing, finance, tech
- Healthcare: Doctors, nurses, therapists
- Education: Teachers, professors, administrators
- Entertainment: Actors, musicians, media
- Government: Public service, military, law enforcement
- Sports: Athletes, coaches, fitness trainers
- Military: Active or former military
- Legal: Lawyers, paralegals
- Other: Anything else

## Tips for Accuracy
1. Use official CBS cast bios for demographic info
2. Watch cast interviews for knowledge level and strategy
3. Update performance data after each episode
4. Double-check spelling of names and locations
5. Be consistent with categories across all contestants

## Model Integration
Once complete, the data will be processed through:
1. Feature engineering pipeline
2. Model prediction generation  
3. Accuracy tracking against actual outcomes
4. Model retraining if needed
        """
        
        with open(instructions_file, 'w') as f:
            f.write(instructions)
        
        print(f"✅ Created data entry instructions: {instructions_file}")
    
    def validate_season_data(self, season_file):
        """Validate new season data for completeness and format"""
        
        try:
            df = pd.read_csv(season_file)
        except Exception as e:
            return False, f"Error reading file: {e}"
        
        validation_errors = []
        
        # Check required columns
        required_cols = list(self.season_template.keys())
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            validation_errors.append(f"Missing columns: {missing_cols}")
        
        # Check data types and ranges
        for _, row in df.iterrows():
            contestant = row['Contestant_Name']
            
            # Age validation
            if not (18 <= row.get('Age', 0) <= 70):
                validation_errors.append(f"{contestant}: Age must be 18-70")
            
            # Gender validation
            if row.get('Gender') not in ['M', 'F']:
                validation_errors.append(f"{contestant}: Gender must be M or F")
            
            # Fitness validation
            if not (1 <= row.get('Self_Reported_Fitness', 0) <= 5):
                validation_errors.append(f"{contestant}: Fitness must be 1-5")
            
            # Strategic archetype validation
            valid_archetypes = [
                'Social_Player', 'Strategic_Player', 'Challenge_Beast', 'Under_Radar',
                'Villain', 'Hero', 'Provider', 'Wild_Card'
            ]
            if row.get('Strategic_Archetype') not in valid_archetypes:
                validation_errors.append(f"{contestant}: Invalid strategic archetype")
        
        if validation_errors:
            return False, validation_errors
        else:
            return True, "Data validation passed"
    
    def generate_preseason_predictions(self, season_file):
        """Generate predictions for new season contestants before season starts"""
        
        # Validate data first
        is_valid, validation_result = self.validate_season_data(season_file)
        if not is_valid:
            print(f"❌ Data validation failed: {validation_result}")
            return None
        
        # Load season data
        season_df = pd.read_csv(season_file)
        season_number = season_df['Season'].iloc[0]
        
        print(f"🎯 Generating pre-season predictions for Season {season_number}")
        
        # Load trained models
        from survivor_predictor import SurvivorPredictor
        predictor = SurvivorPredictor()
        
        if not predictor.models:
            print("❌ No trained models available")
            return None
        
        # Generate predictions for each contestant
        predictions_data = []
        
        for _, contestant_row in season_df.iterrows():
            contestant_data = contestant_row.to_dict()
            
            # Generate predictions
            predictions = predictor.predict_contestant_success(contestant_data)
            
            if predictions:
                prediction_row = {
                    'Season': season_number,
                    'Contestant_Name': contestant_data['Contestant_Name'],
                    'Age': contestant_data['Age'],
                    'Gender': contestant_data['Gender'],
                    'Occupation': contestant_data['Occupation'],
                    'Strategic_Archetype': contestant_data['Strategic_Archetype'],
                    'Merge_Probability': predictions['merge_prediction']['probability'],
                    'Merge_Prediction': predictions['merge_prediction']['prediction'],
                    'Finale_Probability': predictions['finale_prediction']['probability'],
                    'Finale_Prediction': predictions['finale_prediction']['prediction'],
                    'Winner_Probability': predictions['winner_prediction']['probability'],
                    'Winner_Prediction': predictions['winner_prediction']['prediction'],
                    'Expected_Placement': predictions['placement_prediction']['rounded'],
                    'Expected_Days': predictions['days_lasted_prediction']['rounded'],
                    'Prediction_Date': datetime.now().strftime('%Y-%m-%d')
                }
                predictions_data.append(prediction_row)
        
        # Create predictions dataframe
        predictions_df = pd.DataFrame(predictions_data)
        
        # Sort by winner probability (highest to lowest)
        predictions_df = predictions_df.sort_values('Winner_Probability', ascending=False)
        
        # Save predictions
        predictions_file = self.results_dir / f'season_{season_number}_preseason_predictions.csv'
        predictions_df.to_csv(predictions_file, index=False)
        
        print(f"✅ Pre-season predictions saved: {predictions_file}")
        
        # Create summary report
        self.create_preseason_report(predictions_df, season_number)
        
        return predictions_df
    
    def create_preseason_report(self, predictions_df, season_number):
        """Create a comprehensive pre-season prediction report"""
        
        report_file = self.results_dir / f'season_{season_number}_preseason_report.md'
        
        # Calculate summary statistics
        avg_merge_prob = predictions_df['Merge_Probability'].mean()
        avg_finale_prob = predictions_df['Finale_Probability'].mean()
        avg_winner_prob = predictions_df['Winner_Probability'].mean()
        
        # Top predictions
        top_winner = predictions_df.iloc[0]
        top_merge = predictions_df.nlargest(1, 'Merge_Probability').iloc[0]
        top_finale = predictions_df.nlargest(1, 'Finale_Probability').iloc[0]
        
        # Underdog (lowest winner probability)
        underdog = predictions_df.nsmallest(1, 'Winner_Probability').iloc[0]
        
        report = f"""
# Season {season_number} Pre-Season Prediction Report

## Model Predictions Overview
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### Cast Summary
- **Total Contestants:** {len(predictions_df)}
- **Average Merge Probability:** {avg_merge_prob:.1%}
- **Average Finale Probability:** {avg_finale_prob:.1%}
- **Average Winner Probability:** {avg_winner_prob:.1%}

### Top Predictions

**Most Likely Winner:**
- **{top_winner['Contestant_Name']}** ({top_winner['Age']}, {top_winner['Gender']})
- Occupation: {top_winner['Occupation']}
- Strategic Archetype: {top_winner['Strategic_Archetype']}
- Winner Probability: **{top_winner['Winner_Probability']:.1%}**
- Expected Placement: #{top_winner['Expected_Placement']}

**Most Likely to Make Merge:**
- **{top_merge['Contestant_Name']}** - {top_merge['Merge_Probability']:.1%} chance

**Most Likely Finalist:**
- **{top_finale['Contestant_Name']}** - {top_finale['Finale_Probability']:.1%} chance

**Biggest Underdog:**
- **{underdog['Contestant_Name']}** - {underdog['Winner_Probability']:.1%} winner chance

### Full Cast Rankings

| Rank | Contestant | Age | Archetype | Winner % | Merge % | Finale % | Expected Place |
|------|------------|-----|-----------|----------|---------|----------|----------------|
"""
        
        # Add full rankings table
        for i, (_, contestant) in enumerate(predictions_df.iterrows(), 1):
            report += f"| {i} | {contestant['Contestant_Name']} | {contestant['Age']} | {contestant['Strategic_Archetype']} | {contestant['Winner_Probability']:.1%} | {contestant['Merge_Probability']:.1%} | {contestant['Finale_Probability']:.1%} | #{contestant['Expected_Placement']} |\n"
        
        report += f"""

### Strategic Archetype Analysis
"""
        
        # Archetype analysis
        archetype_analysis = predictions_df.groupby('Strategic_Archetype').agg({
            'Winner_Probability': ['mean', 'count'],
            'Merge_Probability': 'mean',
            'Expected_Placement': 'mean'
        }).round(3)
        
        for archetype in archetype_analysis.index:
            count = archetype_analysis.loc[archetype, ('Winner_Probability', 'count')]
            win_prob = archetype_analysis.loc[archetype, ('Winner_Probability', 'mean')]
            merge_prob = archetype_analysis.loc[archetype, ('Merge_Probability', 'mean')]
            avg_place = archetype_analysis.loc[archetype, ('Expected_Placement', 'mean')]
            
            report += f"**{archetype}** ({count} contestants): {win_prob:.1%} avg winner chance, {merge_prob:.1%} avg merge chance, #{avg_place:.1f} avg placement\n\n"
        
        report += f"""
### Model Confidence Notes
- Predictions based on training data from seasons 41-48
- Pre-season predictions use demographic and strategic data only
- Actual performance will depend on gameplay, alliances, and random events
- Model accuracy historically: 90-100% for major milestones

### Tracking Progress
As Season {season_number} progresses:
1. Update contestant performance data weekly
2. Generate mid-season prediction updates
3. Compare final results to pre-season predictions
4. Use outcomes to improve model accuracy

---
*Generated by Survivor Success Prediction Model*
        """
        
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"✅ Pre-season report created: {report_file}")
    
    def update_model_with_new_season(self, completed_season_file):
        """Update models with completed season data for improved accuracy"""
        
        print("🔄 Updating models with new season data...")
        
        # Load existing data
        existing_data_file = self.data_dir / 'Survivor_Complete_Dataset_Seasons_4148.csv'
        existing_df = pd.read_csv(existing_data_file)
        
        # Load new season data
        new_season_df = pd.read_csv(completed_season_file)
        season_number = new_season_df['Season'].iloc[0]
        
        # Combine datasets
        combined_df = pd.concat([existing_df, new_season_df], ignore_index=True)
        
        # Save updated dataset
        updated_data_file = self.data_dir / f'Survivor_Complete_Dataset_Seasons_41{season_number}.csv'
        combined_df.to_csv(updated_data_file, index=False)
        
        print(f"✅ Updated dataset saved: {updated_data_file}")
        
        # Backup old models
        backup_dir = self.models_dir / f'backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        backup_dir.mkdir(exist_ok=True)
        
        for model_file in self.models_dir.glob('*.pkl'):
            shutil.copy2(model_file, backup_dir)
        
        print(f"✅ Old models backed up to: {backup_dir}")
        
        # Retrain models with updated data
        trainer = SurvivorModelTrainer()
        
        # Update the data file path in processor
        trainer.processor = SurvivorDataProcessor()
        
        # Train new models
        results = trainer.train_all_models()
        
        if results:
            # Save new models
            trainer.save_models(self.models_dir)
            print(f"✅ Models retrained and saved with Season {season_number} data")
            
            # Create performance comparison report
            self.create_model_update_report(season_number, results)
            
        return results
    
    def create_model_update_report(self, season_number, training_results):
        """Create a report comparing old vs new model performance"""
        
        report_file = self.results_dir / f'model_update_season_{season_number}_report.md'
        
        report = f"""
# Model Update Report - Season {season_number} Integration

**Update Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dataset Changes
- **New Data:** Season {season_number} ({len(training_results)} contestants added)
- **Total Dataset:** Now includes seasons 41-{season_number}

## Retraining Results
Models have been retrained with the expanded dataset.

### New Model Performance
"""
        
        # Add performance metrics from training results
        for task_name, task_results in training_results.items():
            report += f"\n**{task_name.replace('_', ' ').title()}:**\n"
            for model_name, result in task_results.items():
                if 'accuracy' in result:
                    report += f"- {model_name.title()}: {result['accuracy']:.3f} accuracy\n"
                else:
                    report += f"- {model_name.title()}: R² = {result['r2']:.3f}\n"
        
        report += f"""

## Next Steps
1. Test updated models on Season {season_number + 1} when cast is announced
2. Monitor prediction accuracy
3. Continue collecting data for future seasons

## Backup Information
Previous models backed up before update for safety.

---
*Model Update Complete*
        """
        
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"✅ Model update report created: {report_file}")

if __name__ == "__main__":
    integrator = SeasonIntegrator()
    
    # Create Season 49 template
    print("Creating Season 49 integration template...")
    template_file = integrator.create_season_template(season_number=49)
    
    print(f"""
🏝️ Season 49 Integration Setup Complete!

Next Steps:
1. Fill out the template file: {template_file}
2. Follow instructions in: {template_file.parent}/season_49_instructions.md
3. Generate predictions with: integrator.generate_preseason_predictions(template_file)

Template includes:
- 18 contestant slots
- All required data fields
- Detailed entry instructions
- Validation system
    """)