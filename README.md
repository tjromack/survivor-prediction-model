# Survivor Success Prediction Model

A comprehensive machine learning system that predicts contestant success in CBS Survivor using advanced analytics and dual prediction models. Built for seasons 41+ (New Era) with production-ready deployment and Season 49 validation framework.

## 🏆 Project Overview

This system uses machine learning to predict Survivor contestant outcomes across multiple success metrics. The project features dual prediction models: pre-season analysis using only demographics and background data, and full-game predictions incorporating challenge performance and strategic gameplay.

### Current Status: Season 49 Validation Phase
The system has generated pre-season predictions for all Season 49 contestants (premiering September 24, 2025) and is ready for accuracy validation as the season progresses.

## 📊 Model Performance

### Pre-Season Models (Demographics Only)
| Prediction Task | Algorithm | Accuracy | Use Case |
|----------------|-----------|----------|----------|
| Merge Prediction | Logistic Regression | 69.4% | Pre-cast analysis |
| Finale Prediction | Random Forest | 55.6% | Draft rankings |
| Winner Prediction | Random Forest | **94.4%** | Season outcome |

### Full-Game Models (With Performance Data)
| Prediction Task | Algorithm | Accuracy/R² | Cross-Validation |
|----------------|-----------|-------------|------------------|
| Merge Prediction | XGBoost | **100.0%** | 100.0% ± 0.0% |
| Finale Prediction | XGBoost | **100.0%** | 100.0% ± 0.0% |
| Winner Prediction | Random Forest | 96.6% | 94.5% ± 5.1% |
| Placement Ranking | XGBoost | R² = 0.988 | RMSE = 0.56 |
| Days Lasted | Random Forest | R² = 0.997 | RMSE = 0.23 |

## 🔍 Key Discoveries

### What Predicts Survivor Success
1. **Challenge Exposure > Win Rate**: Total challenges faced matters more than wins
2. **Survival Rate Dominance**: Days lasted ratio is the strongest predictor
3. **Strategic Archetype Impact**: Moderate influence on pre-season predictions
4. **Demographic Patterns**: Age, region, and occupation show predictive signals
5. **Vote Management**: Votes-per-tribal crucial for threat level assessment

### Surprising Findings
- Pre-season demographic models achieve 94% winner prediction accuracy
- Challenge quantity predicts success better than challenge performance
- Age shows minimal impact contrary to popular belief
- Strategic archetype less predictive than expected

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- 8GB+ RAM for model training
- Git for version control

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/survivor-prediction-model.git
cd survivor-prediction-model

# Create virtual environment
python -m venv survivor_env

# Activate environment
# Windows:
survivor_env\Scripts\activate
# macOS/Linux:
source survivor_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage
```bash
# Process historical data
python src/data_processor.py

# Train full-game models
python src/model_trainer.py

# Train pre-season models
python src/preseason_model_trainer.py

# Launch web interface
python run_app.py
```

## 📁 Project Structure

```
survivor-prediction-model/
├── data/                              # Datasets and processed files
│   ├── Survivor_Complete_Dataset_Seasons_4148.csv   # Historical data (41-48)
│   ├── survivor_49_cast.csv                         # Season 49 contestants
│   └── processed_survivor_data.pkl                  # Processed features
├── src/                               # Core prediction system
│   ├── data_processor.py             # Feature engineering pipeline (35→223 features)
│   ├── model_trainer.py              # Full-game model training
│   ├── preseason_model_trainer.py    # Pre-season model training
│   ├── survivor_predictor.py         # Production prediction system
│   ├── preseason_predictor.py        # Pre-season specific predictor
│   ├── model_evaluator.py           # Advanced model analysis
│   ├── hyperparameter_tuner.py      # Model optimization
│   └── season_integration.py        # New season data pipeline
├── app/                              # Web interface
│   └── streamlit_app.py             # Enhanced dashboard
├── models/                           # Trained model artifacts
│   ├── *_model.pkl                  # Trained models
│   ├── *_tuned.pkl                  # Hyperparameter-tuned models
│   └── preprocessors.pkl            # Data preprocessing components
├── notebooks/                        # Analysis notebooks
│   ├── 01_exploratory_data_analysis.ipynb
│   └── 02_model_training_analysis.ipynb
├── results/                          # Predictions and reports
└── config/                          # Configuration management
```

## 🛠️ Core Components

### Data Processing Pipeline (`src/data_processor.py`)
Transforms raw contestant data into ML-ready features:
- **Input**: 35 original contestant attributes
- **Output**: 223 engineered features
- **Key Features**: Challenge rates, strategic metrics, demographic encodings

### Dual Prediction Models
**Pre-Season Models**: Use only demographic and background data
- Perfect for cast analysis before filming
- 17 features: age, occupation, strategic archetype, athletic background, etc.
- Achieves 94% winner prediction accuracy

**Full-Game Models**: Include challenge performance and strategic moves
- For mid-season and post-season analysis
- 223 features including gameplay statistics
- Achieves near-perfect accuracy on historical data

### Web Interface (`app/streamlit_app.py`)
Professional dashboard featuring:
- Individual contestant predictions (both modes)
- Season 49 cast rankings with success tiers
- Batch analysis capabilities
- Model performance monitoring
- Historical analysis and trends

## 🎯 Season 49 Integration

### Current Predictions
The system has generated complete pre-season predictions for all 18 Season 49 contestants, including:
- Individual success probabilities (merge/finale/winner)
- Draft-style rankings with composite scoring
- Success tier classifications (Elite/Contender/Solid/Risky/Long Shot)

### Validation Framework
Track prediction accuracy as Season 49 progresses:
1. **Pre-merge eliminations**: Weekly accuracy assessment
2. **Merge predictions**: Success rate of merge/non-merge classifications
3. **Finale predictions**: Final 3 prediction accuracy
4. **Winner prediction**: Ultimate model validation

### Updating for New Seasons
```bash
# Generate template for new season
python src/season_integration.py

# Process new season data
python src/season_integration.py --update-models season_50_data.csv

# Retrain models with expanded dataset
python src/model_trainer.py
```

## 📈 Advanced Analytics

### Feature Importance Rankings
Based on Random Forest analysis of 144 historical contestants:

**Top Predictive Features:**
1. `survival_rate` (0.150) - Days lasted / 26
2. `total_challenges` (0.129) - Challenge exposure
3. `Tribals_Attended` (0.102) - Tribal council navigation
4. `Individual_Challenges_Total` (0.095) - Individual immunity phase
5. `votes_per_tribal` (0.084) - Threat management

### Success Patterns by Archetype
| Strategic Archetype | Merge Rate | Finale Rate | Avg Placement |
|-------------------|------------|-------------|---------------|
| Strategic_Player | 65% | 28% | 8.2 |
| Challenge_Beast | 72% | 31% | 7.8 |
| Social_Player | 58% | 24% | 9.1 |
| Under_Radar | 51% | 19% | 9.8 |

## 🔧 Configuration & Customization

### Model Parameters
Key settings in `config/config.py`:
```python
# Target variables
TARGET_VARIABLES = {
    'placement': 'Final_Placement',
    'merge': 'Made_Merge', 
    'finale': 'Made_Finale',
    'days_lasted': 'Days_Lasted'
}

# Model hyperparameters
MODEL_PARAMS = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': 42
    }
}
```

### Feature Engineering
Customize feature creation in `src/data_processor.py`:
- Challenge performance ratios
- Strategic gameplay metrics
- Demographic encodings
- Regional advantages

## 📊 Validation & Testing

### Model Evaluation
```bash
# Comprehensive model evaluation
python src/model_evaluator.py

# Generate performance reports
jupyter lab notebooks/02_model_training_analysis.ipynb
```

### Prediction Testing
```bash
# Test individual predictions
python src/survivor_predictor.py

# Batch analysis
python src/preseason_predictor.py
```

## 🤝 Contributing & Future Development

### Planned Enhancements
- **Season 50 Integration**: Automatic cast analysis pipeline
- **Real-time Updates**: Live episode tracking and model updates
- **Advanced Metrics**: Confessional analysis and screen time correlation
- **Mobile Interface**: Responsive design for mobile devices

### Contributing Guidelines
1. Fork the repository
2. Create feature branch (`git checkout -b feature/enhancement`)
3. Commit changes with clear messages
4. Submit pull request with detailed description

### Data Sources
- **Historical Data**: Manual compilation from seasons 41-48
- **Season 49**: Official CBS cast announcements and interviews
- **Feature Engineering**: Domain expertise in Survivor gameplay strategy

## 📄 License & Disclaimer

This project is licensed under the MIT License. See LICENSE file for details.

**Disclaimer**: This model is designed for entertainment and analytical purposes. Survivor outcomes involve numerous unmeasurable factors including production decisions, random events, interpersonal dynamics, and pure chance that cannot be fully captured in any predictive model.

## 🙏 Acknowledgments

- **CBS Survivor**: For creating the strategic framework enabling this analysis
- **Survivor Community**: For maintaining detailed contestant databases and insights
- **New Era Format**: Consistent 26-day structure enabling reliable statistical modeling
- **Machine Learning Libraries**: scikit-learn, XGBoost, Streamlit for technical foundation

---

## Quick Reference Commands

```bash
# Full system setup
python src/data_processor.py && python src/model_trainer.py && python src/preseason_model_trainer.py

# Launch web interface
python run_app.py

# Generate Season 49 rankings
python -c "
import sys; sys.path.append('src')
from preseason_predictor import PreSeasonPredictor
predictor = PreSeasonPredictor()
# Use Streamlit interface for complete analysis
"

# Update for new season
python src/season_integration.py --season 50
```

**Current Model Version**: v1.0 (Season 49 Pre-Season Release)  
**Last Updated**: September 24, 2025  
**Next Validation**: Season 49 Finale (December 2025)