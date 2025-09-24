"""
Enhanced Survivor Success Prediction Dashboard
With Pre-Season Models and Cast Rankings
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
import json

# Add src to path - fix for Streamlit
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

# Import with error handling
try:
    from src.survivor_predictor import SurvivorPredictor
    from src.data_processor import SurvivorDataProcessor
    from src.season_integration import SeasonIntegrator
    from src.preseason_predictor import PreSeasonPredictor
    print("✅ Successfully imported all modules")
except ImportError:
    try:
        import survivor_predictor
        import data_processor
        import season_integration
        import preseason_predictor
        SurvivorPredictor = survivor_predictor.SurvivorPredictor
        SurvivorDataProcessor = data_processor.SurvivorDataProcessor
        SeasonIntegrator = season_integration.SeasonIntegrator
        PreSeasonPredictor = preseason_predictor.PreSeasonPredictor
        print("✅ Successfully imported with fallback")
    except ImportError as e:
        st.error(f"❌ Cannot import required modules: {e}")
        st.stop()

# Page configuration
st.set_page_config(
    page_title="Survivor Success Predictor",
    page_icon="🏝️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'predictor' not in st.session_state:
    with st.spinner("Loading models..."):
        st.session_state.predictor = SurvivorPredictor()
        st.session_state.processor = SurvivorDataProcessor()
        st.session_state.integrator = SeasonIntegrator()
        st.session_state.preseason_predictor = PreSeasonPredictor()

# Custom CSS (improved readability)
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    text-align: center;
    color: #1f4e79;
    margin-bottom: 2rem;
}
.success-metric {
    background-color: #ffffff;
    color: #000000;
    padding: 1rem;
    border-radius: 10px;
    border-left: 5px solid #1f4e79;
    margin: 0.5rem 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.prediction-high { 
    color: #155724; 
    background-color: #d4edda;
    padding: 4px 8px;
    border-radius: 4px;
    font-weight: bold; 
}
.prediction-medium { 
    color: #856404; 
    background-color: #fff3cd;
    padding: 4px 8px;
    border-radius: 4px;
    font-weight: bold; 
}
.prediction-low { 
    color: #721c24; 
    background-color: #f8d7da;
    padding: 4px 8px;
    border-radius: 4px;
    font-weight: bold; 
}
.tier-box {
    padding: 1.5rem;
    border-radius: 10px;
    font-size: 1.2rem;
    font-weight: bold;
    text-align: center;
    margin: 1rem 0;
    color: #000000;
    border: 2px solid;
}
.ranking-card {
    background-color: #ffffff;
    border-radius: 10px;
    padding: 1rem;
    margin: 0.5rem 0;
    border-left: 5px solid;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">🏝️ Survivor Success Predictor</h1>', unsafe_allow_html=True)
st.markdown("**Advanced ML predictions for CBS Survivor - Now with Pre-Season Analysis**")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page:", [
    "Individual Prediction", 
    "Season 49 Cast Rankings",
    "Batch Analysis", 
    "Model Performance", 
    "Season 49 Management",
    "Historical Analysis"
])

def create_contestant_input_form():
    """Create the contestant input form"""
    with st.form("contestant_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Demographics")
            name = st.text_input("Contestant Name", value="Test Player")
            age = st.slider("Age", 18, 65, 28)
            gender = st.selectbox("Gender", ["M", "F"])
            home_state = st.selectbox("Home State", [
                "CA", "TX", "NY", "FL", "IL", "PA", "OH", "GA", "NC", "MI",
                "NJ", "VA", "WA", "AZ", "MA", "TN", "IN", "MO", "MD", "WI",
                "CO", "MN", "SC", "AL", "LA", "KY", "OR", "OK", "CT", "UT",
                "IA", "NV", "AR", "MS", "KS", "NM", "NE", "WV", "ID", "HI",
                "NH", "ME", "MT", "RI", "DE", "SD", "ND", "AK", "VT", "WY", "ON"
            ])
            relationship_status = st.selectbox("Relationship Status", [
                "Single", "Married", "In_Relationship", "Divorced", "Engaged"
            ])
        
        with col2:
            st.subheader("Physical & Strategic")
            occupation = st.text_input("Occupation", value="Software Engineer")
            occupation_category = st.selectbox("Occupation Category", [
                "Business", "Healthcare", "Education", "Entertainment", 
                "Government", "Sports", "Military", "Legal", "Other"
            ])
            athletic_background = st.selectbox("Athletic Background", [
                "None", "Recreational", "High_School", "College", "Professional"
            ])
            physical_build = st.selectbox("Physical Build", ["Small", "Medium", "Large"])
            fitness = st.slider("Self-Reported Fitness (1-5)", 1, 5, 4)
            
        with col3:
            st.subheader("Game Knowledge & Strategy")
            survivor_knowledge = st.selectbox("Survivor Knowledge", [
                "No_Knowledge", "Casual_Fan", "Fan", "Superfan"
            ])
            strategic_archetype = st.selectbox("Strategic Archetype", [
                "Social_Player", "Strategic_Player", "Challenge_Beast", 
                "Under_Radar", "Villain", "Hero", "Provider", "Wild_Card"
            ])
            pre_game_target = st.slider("Pre-Game Target Size (1-5)", 1, 5, 3)
        
        # Prediction mode selection
        st.subheader("Prediction Mode")
        prediction_mode = st.radio(
            "Choose prediction type:",
            ["Pre-Season Prediction (Demographics Only)", "Full Game Prediction (Requires Game Stats)"],
            help="Pre-season uses only demographics and background. Full game includes challenge performance and strategic moves."
        )
        
        # Conditional game performance inputs
        if prediction_mode == "Full Game Prediction (Requires Game Stats)":
            st.subheader("Current Season Performance")
            col4, col5 = st.columns(2)
            
            with col4:
                tribal_wins = st.number_input("Tribal Challenges Won", 0, 10, 3)
                tribal_total = st.number_input("Tribal Challenges Total", 0, 15, 5)
                individual_wins = st.number_input("Individual Challenges Won", 0, 10, 2)
                individual_total = st.number_input("Individual Challenges Total", 0, 15, 8)
                
            with col5:
                advantages_found = st.number_input("Advantages Found", 0, 5, 1)
                advantages_played = st.number_input("Advantages Played", 0, 5, 1)
                alliance_count = st.number_input("Alliance Count", 0, 10, 2)
                votes_against = st.number_input("Votes Against Total", 0, 20, 2)
                tribals_attended = st.number_input("Tribals Attended", 0, 15, 8)
        else:
            # Default values for pre-season
            tribal_wins = tribal_total = individual_wins = individual_total = 0
            advantages_found = advantages_played = alliance_count = votes_against = tribals_attended = 0
        
        submitted = st.form_submit_button("Generate Prediction")
        
        return {
            'submitted': submitted,
            'prediction_mode': prediction_mode,
            'contestant_data': {
                'Contestant_Name': name,
                'Age': age,
                'Gender': gender,
                'Home_State': home_state,
                'Home_Region': 'West' if home_state in ['CA', 'WA', 'OR', 'NV', 'AZ', 'UT', 'CO', 'NM', 'WY', 'MT', 'ID', 'AK', 'HI'] else
                              'South' if home_state in ['TX', 'FL', 'GA', 'NC', 'SC', 'VA', 'TN', 'AL', 'LA', 'KY', 'AR', 'MS', 'WV', 'OK', 'MD', 'DE'] else
                              'Midwest' if home_state in ['IL', 'IN', 'OH', 'MI', 'WI', 'MN', 'IA', 'MO', 'ND', 'SD', 'NE', 'KS'] else 'Northeast',
                'Relationship_Status': relationship_status,
                'Occupation': occupation,
                'Occupation_Category': occupation_category,
                'Athletic_Background': athletic_background,
                'Physical_Build': physical_build,
                'Self_Reported_Fitness': fitness,
                'Survivor_Knowledge': survivor_knowledge,
                'Strategic_Archetype': strategic_archetype,
                'Pre_Game_Target_Size': pre_game_target,
                'Tribal_Challenges_Won': tribal_wins,
                'Tribal_Challenges_Total': tribal_total,
                'Individual_Challenges_Won': individual_wins,
                'Individual_Challenges_Total': individual_total,
                'Advantages_Found': advantages_found,
                'Advantages_Played': advantages_played,
                'Alliance_Count': alliance_count,
                'Votes_Against_Total': votes_against,
                'Tribals_Attended': tribals_attended,
                # Add dummy values for processing
                'Made_Merge': 'Y',
                'Made_Finale': 'N',
                'Final_Placement': 10,
                'Days_Lasted': 20,
                'Elimination_Type': 'Voted_Out',
                'Jury_Votes_Received': 0
            }
        }

def display_prediction_results(predictions, prediction_mode, contestant_name):
    """Display prediction results with improved styling"""
    st.success(f"Predictions Generated for {contestant_name}!")
    
    # Add mode indicator
    mode_color = "#17a2b8" if "Pre-Season" in prediction_mode else "#28a745"
    st.markdown(f'<div style="background-color: {mode_color}20; padding: 0.5rem; border-radius: 5px; text-align: center; margin-bottom: 1rem;"><strong>Mode: {prediction_mode}</strong></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Success Probabilities")
        
        # Merge prediction
        merge_prob = predictions['merge_prediction']['probability']
        merge_color = "prediction-high" if merge_prob > 0.7 else "prediction-medium" if merge_prob > 0.4 else "prediction-low"
        st.markdown(f'<div class="success-metric">🎯 <strong>Make Merge:</strong> <span class="{merge_color}">{merge_prob:.1%}</span></div>', unsafe_allow_html=True)
        
        # Finale prediction
        finale_prob = predictions['finale_prediction']['probability']
        finale_color = "prediction-high" if finale_prob > 0.5 else "prediction-medium" if finale_prob > 0.2 else "prediction-low"
        st.markdown(f'<div class="success-metric">🏆 <strong>Make Finale:</strong> <span class="{finale_color}">{finale_prob:.1%}</span></div>', unsafe_allow_html=True)
        
        # Winner prediction
        winner_prob = predictions['winner_prediction']['probability']
        winner_color = "prediction-high" if winner_prob > 0.15 else "prediction-medium" if winner_prob > 0.08 else "prediction-low"
        st.markdown(f'<div class="success-metric">👑 <strong>Win Game:</strong> <span class="{winner_color}">{winner_prob:.1%}</span></div>', unsafe_allow_html=True)
        
        # Additional metrics for full game predictions
        if "Full Game" in prediction_mode:
            placement = predictions['placement_prediction']['rounded']
            st.markdown(f'<div class="success-metric">🥇 <strong>Expected Placement:</strong> #{placement}</div>', unsafe_allow_html=True)
            
            days = predictions['days_lasted_prediction']['rounded']
            st.markdown(f'<div class="success-metric">📅 <strong>Expected Days Lasted:</strong> {days} days</div>', unsafe_allow_html=True)
    
    with col2:
        st.subheader("Success Probability Chart")
        
        probs = [merge_prob, finale_prob, winner_prob]
        labels = ['Make Merge', 'Make Finale', 'Win Game']
        
        fig = px.bar(
            x=labels, y=probs,
            title="Success Probabilities",
            color=probs,
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(
            showlegend=False,
            yaxis_title="Probability",
            xaxis_title=""
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Success tier assessment
    st.subheader("Success Tier Assessment")
    if winner_prob > 0.15:
        tier = "🏆 WINNER POTENTIAL - High chance of victory!"
        tier_color = "#28a745"
    elif finale_prob > 0.5:
        tier = "🥈 FINALIST MATERIAL - Likely to make final 3"
        tier_color = "#17a2b8"
    elif merge_prob > 0.7:
        tier = "⭐ STRONG PLAYER - Should make jury"
        tier_color = "#28a745"
    elif merge_prob > 0.4:
        tier = "📈 DECENT SHOT - 50/50 merge chances"
        tier_color = "#ffc107"
    else:
        tier = "⚠️ EARLY BOOT RISK - Needs strong strategy"
        tier_color = "#dc3545"
    
    st.markdown(f'''
    <div class="tier-box" style="background-color: {tier_color}30; border-color: {tier_color};">
        <span style="color: #000000;">{tier}</span>
    </div>
    ''', unsafe_allow_html=True)

def create_season49_rankings():
    """Create comprehensive Season 49 cast rankings"""
    st.header("Season 49 Cast Rankings & Analysis")
    st.markdown("**Draft-style rankings based on pre-season predictions and multiple success factors**")
    
    # Load Season 49 data
    season49_file = project_root / 'data' / 'survivor_49_cast.csv'
    
    if not season49_file.exists():
        st.warning("Season 49 cast data not found. Please upload the cast CSV first.")
        
        uploaded_file = st.file_uploader("Upload Season 49 Cast CSV", type=['csv'])
        if uploaded_file:
            season49_df = pd.read_csv(uploaded_file)
            season49_df.to_csv(season49_file, index=False)
            st.success("Season 49 cast data uploaded!")
        else:
            return
    else:
        season49_df = pd.read_csv(season49_file)
    
    st.subheader("Cast Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Contestants", len(season49_df))
    with col2:
        st.metric("Age Range", f"{season49_df['Age'].min()}-{season49_df['Age'].max()}")
    with col3:
        st.metric("Strategic Archetypes", season49_df['Strategic_Archetype'].nunique())
    with col4:
        st.metric("Occupation Categories", season49_df['Occupation_Category'].nunique())
    
    # Generate rankings
    if st.button("Generate Complete Cast Rankings", type="primary"):
        with st.spinner("Analyzing entire Season 49 cast..."):
            rankings_data = []
            
            for _, contestant in season49_df.iterrows():
                contestant_dict = contestant.to_dict()
                
                # Get pre-season predictions
                predictions = st.session_state.preseason_predictor.predict_contestant_preseason(contestant_dict)
                
                if predictions:
                    # Calculate composite scores
                    merge_prob = predictions.get('merge_prediction', {}).get('probability', 0)
                    finale_prob = predictions.get('finale_prediction', {}).get('probability', 0)
                    winner_prob = predictions.get('winner_prediction', {}).get('probability', 0)
                    
                    # Weighted composite score (emphasizing winner potential)
                    composite_score = (winner_prob * 0.5) + (finale_prob * 0.3) + (merge_prob * 0.2)
                    
                    # Success tier
                    if winner_prob > 0.15:
                        tier = "Elite"
                    elif finale_prob > 0.5:
                        tier = "Contender"
                    elif merge_prob > 0.7:
                        tier = "Solid"
                    elif merge_prob > 0.4:
                        tier = "Risky"
                    else:
                        tier = "Long Shot"
                    
                    rankings_data.append({
                        'Rank': 0,  # Will be filled after sorting
                        'Name': contestant_dict['Contestant_Name'],
                        'Age': contestant_dict['Age'],
                        'Occupation': contestant_dict['Occupation'],
                        'Archetype': contestant_dict['Strategic_Archetype'],
                        'Merge_Prob': merge_prob,
                        'Finale_Prob': finale_prob,
                        'Winner_Prob': winner_prob,
                        'Composite_Score': composite_score,
                        'Tier': tier,
                        'Knowledge': contestant_dict['Survivor_Knowledge'],
                        'Athletic_Background': contestant_dict['Athletic_Background']
                    })
            
            # Sort by composite score and assign ranks
            rankings_df = pd.DataFrame(rankings_data)
            rankings_df = rankings_df.sort_values('Composite_Score', ascending=False).reset_index(drop=True)
            rankings_df['Rank'] = rankings_df.index + 1
            
            # Store in session state
            st.session_state.season49_rankings = rankings_df
    
    # Display rankings if available
    if 'season49_rankings' in st.session_state:
        rankings_df = st.session_state.season49_rankings
        
        # Ranking visualization tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Overall Rankings", "Winner Probabilities", "Tier Analysis", "Detailed Stats"])
        
        with tab1:
            st.subheader("Complete Cast Rankings")
            
            # Top 10 with enhanced display
            st.markdown("### 🏆 Top 10 Draft Picks")
            top_10 = rankings_df.head(10)
            
            for i, (_, contestant) in enumerate(top_10.iterrows()):
                # Tier colors
                tier_colors = {
                    'Elite': '#28a745',
                    'Contender': '#17a2b8', 
                    'Solid': '#ffc107',
                    'Risky': '#fd7e14',
                    'Long Shot': '#dc3545'
                }
                
                tier_color = tier_colors.get(contestant['Tier'], '#6c757d')
                
                st.markdown(f'''
                <div class="ranking-card" style="border-left-color: {tier_color};">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <h4 style="margin: 0; color: #000;">#{contestant['Rank']} {contestant['Name']}</h4>
                            <p style="margin: 5px 0; color: #666;">
                                {contestant['Age']} • {contestant['Occupation']} • {contestant['Archetype']}
                            </p>
                        </div>
                        <div style="text-align: right;">
                            <div style="background-color: {tier_color}; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; margin-bottom: 4px;">
                                {contestant['Tier']}
                            </div>
                            <small style="color: #666;">Winner: {contestant['Winner_Prob']:.1%}</small>
                        </div>
                    </div>
                </div>
                ''', unsafe_allow_html=True)
            
            # Full rankings table
            st.subheader("Complete Rankings Table")
            display_df = rankings_df[['Rank', 'Name', 'Age', 'Archetype', 'Tier', 'Winner_Prob', 'Finale_Prob', 'Merge_Prob']].copy()
            display_df['Winner_Prob'] = display_df['Winner_Prob'].apply(lambda x: f"{x:.1%}")
            display_df['Finale_Prob'] = display_df['Finale_Prob'].apply(lambda x: f"{x:.1%}")
            display_df['Merge_Prob'] = display_df['Merge_Prob'].apply(lambda x: f"{x:.1%}")
            
            st.dataframe(display_df, use_container_width=True)
        
        with tab2:
            st.subheader("Winner Probability Rankings")
            
            fig = px.bar(
                rankings_df.head(15),
                x='Winner_Prob',
                y='Name',
                orientation='h',
                title="Top 15 Winner Probabilities",
                color='Winner_Prob',
                color_continuous_scale='RdYlGn'
            )
            fig.update_layout(height=600, yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Success Tier Analysis")
            
            tier_counts = rankings_df['Tier'].value_counts()
            
            col1, col2 = st.columns(2)
            with col1:
                fig_pie = px.pie(
                    values=tier_counts.values,
                    names=tier_counts.index,
                    title="Cast Distribution by Success Tier"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                st.markdown("### Tier Definitions")
                st.markdown("""
                - **Elite**: Winner probability > 15%
                - **Contender**: Finale probability > 50%  
                - **Solid**: Merge probability > 70%
                - **Risky**: Merge probability 40-70%
                - **Long Shot**: Merge probability < 40%
                """)
        
        with tab4:
            st.subheader("Detailed Statistical Analysis")
            
            # Archetype analysis
            archetype_stats = rankings_df.groupby('Archetype').agg({
                'Winner_Prob': ['mean', 'count'],
                'Finale_Prob': 'mean',
                'Merge_Prob': 'mean'
            }).round(3)
            
            archetype_stats.columns = ['Avg_Winner_Prob', 'Count', 'Avg_Finale_Prob', 'Avg_Merge_Prob']
            st.markdown("### Success by Strategic Archetype")
            st.dataframe(archetype_stats)
            
            # Age analysis
            age_bins = pd.cut(rankings_df['Age'], bins=[20, 30, 40, 55], labels=['20-30', '31-40', '41+'])
            age_stats = rankings_df.groupby(age_bins).agg({
                'Winner_Prob': 'mean',
                'Finale_Prob': 'mean',
                'Merge_Prob': 'mean'
            }).round(3)
            
            st.markdown("### Success by Age Group")
            st.dataframe(age_stats)
        
        # Download functionality
        st.markdown("---")
        csv = rankings_df.to_csv(index=False)
        st.download_button(
            label="📊 Download Complete Rankings",
            data=csv,
            file_name="season49_cast_rankings.csv",
            mime="text/csv"
        )

# Main page routing
if page == "Individual Prediction":
    st.header("Individual Contestant Prediction")
    
    form_data = create_contestant_input_form()
    
    if form_data['submitted']:
        contestant_data = form_data['contestant_data']
        prediction_mode = form_data['prediction_mode']
        
        with st.spinner("Generating predictions..."):
            if "Pre-Season" in prediction_mode:
                predictions = st.session_state.preseason_predictor.predict_contestant_preseason(contestant_data)
            else:
                predictions = st.session_state.predictor.predict_contestant_success(contestant_data)
        
        if predictions:
            display_prediction_results(predictions, prediction_mode, contestant_data['Contestant_Name'])
        else:
            st.error("Unable to generate predictions. Please check your inputs.")

elif page == "Season 49 Cast Rankings":
    create_season49_rankings()

elif page == "Season 49 Management":
    st.header("Season 49 Cast Management")
    
    integrator = st.session_state.integrator
    
    # Template creation
    st.subheader("Create Season Template")
    col1, col2 = st.columns(2)
    
    with col1:
        season_num = st.number_input("Season Number", 49, 60, 49)
        num_contestants = st.number_input("Number of Contestants", 16, 20, 18)
    
    with col2:
        if st.button("Create Template"):
            template_file = integrator.create_season_template(season_num, num_contestants)
            st.success(f"Template created: {template_file.name}")
            st.info("Check the data/ directory for the template CSV and instructions")
    
    # File upload and validation
    st.subheader("Upload Season Data")
    uploaded_file = st.file_uploader("Upload completed season CSV", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Save uploaded file temporarily
            temp_file = integrator.data_dir / f"temp_{uploaded_file.name}"
            with open(temp_file, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            
            # Validate data
            is_valid, validation_result = integrator.validate_season_data(temp_file)
            
            if is_valid:
                st.success("Data validation passed!")
                
                # Load and display data
                season_df = pd.read_csv(temp_file)
                st.subheader("Cast Overview")
                st.dataframe(season_df[['Contestant_Name', 'Age', 'Gender', 'Occupation', 'Strategic_Archetype']].head(10))
                
                # Generate predictions button
                if st.button("Generate Pre-Season Predictions"):
                    with st.spinner("Generating predictions..."):
                        predictions_df = integrator.generate_preseason_predictions(temp_file)
                    
                    if predictions_df is not None:
                        st.success("Predictions generated!")
                        
                        # Display top predictions
                        st.subheader("Winner Predictions (Top 5)")
                        top_5 = predictions_df.head(5)[['Contestant_Name', 'Winner_Probability', 'Merge_Probability', 'Expected_Placement']]
                        st.dataframe(top_5)
                        
                        # Create visualization
                        fig = px.bar(
                            predictions_df.head(10),
                            x='Contestant_Name',
                            y='Winner_Probability',
                            title="Top 10 Winner Probabilities"
                        )
                        fig.update_xaxes(tickangle=45)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Download predictions
                        csv = predictions_df.to_csv(index=False)
                        st.download_button(
                            "Download Full Predictions",
                            csv,
                            f"season_{season_df['Season'].iloc[0]}_predictions.csv",
                            "text/csv"
                        )
            else:
                st.error("Data validation failed!")
                st.write("Validation errors:")
                for error in validation_result:
                    st.write(f"- {error}")
            
            # Clean up temp file
            if temp_file.exists():
                temp_file.unlink()
            
        except Exception as e:
            st.error(f"Error processing file: {e}")

elif page == "Batch Analysis":
    st.header("Batch Analysis")
    st.info("Upload multiple contestants for batch predictions")
    
    # File upload for batch analysis
    uploaded_file = st.file_uploader("Upload CSV file with contestants", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            # Prediction mode selection
            batch_mode = st.radio(
                "Batch Prediction Mode:",
                ["Pre-Season Predictions", "Full Game Predictions"]
            )
            
            if st.button("Run Batch Predictions"):
                with st.spinner("Processing batch predictions..."):
                    batch_predictions = []
                    
                    for _, row in df.iterrows():
                        contestant_data = row.to_dict()
                        
                        # Add dummy values if missing
                        if 'Made_Merge' not in contestant_data:
                            contestant_data.update({
                                'Made_Merge': 'Y', 'Made_Finale': 'N', 'Final_Placement': 10,
                                'Days_Lasted': 20, 'Elimination_Type': 'Voted_Out',
                                'Jury_Votes_Received': 0
                            })
                        
                        # Choose predictor based on mode
                        if batch_mode == "Pre-Season Predictions":
                            predictions = st.session_state.preseason_predictor.predict_contestant_preseason(contestant_data)
                        else:
                            predictions = st.session_state.predictor.predict_contestant_success(contestant_data)
                        
                        if predictions:
                            batch_predictions.append({
                                'Name': contestant_data.get('Contestant_Name', 'Unknown'),
                                'Merge_Prob': predictions['merge_prediction']['probability'],
                                'Finale_Prob': predictions['finale_prediction']['probability'],
                                'Winner_Prob': predictions['winner_prediction']['probability']
                            })
                
                if batch_predictions:
                    results_df = pd.DataFrame(batch_predictions)
                    results_df = results_df.sort_values('Winner_Prob', ascending=False)
                    
                    st.subheader("Batch Prediction Results")
                    st.dataframe(results_df)
                    
                    # Visualization
                    fig = px.bar(results_df.head(10), x='Name', y='Winner_Prob', 
                               title="Winner Probabilities")
                    fig.update_xaxes(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button("Download Results", csv, f"batch_predictions_{batch_mode.lower().replace(' ', '_')}.csv", "text/csv")
        
        except Exception as e:
            st.error(f"Error processing batch file: {e}")

elif page == "Model Performance":
    st.header("Model Performance Dashboard")
    
    # Load training results if available
    try:
        import joblib
        results_file = project_root / 'models' / 'training_results.pkl'
        if results_file.exists():
            results = joblib.load(results_file)
            
            st.subheader("Model Accuracy Comparison")
            
            # Create performance comparison
            comparison_data = []
            for task_name, task_results in results.items():
                for model_name, result in task_results.items():
                    if 'accuracy' in result:
                        comparison_data.append({
                            'Task': task_name.replace('_', ' ').title(),
                            'Model': model_name.title(),
                            'Accuracy': result['accuracy'],
                            'CV Score': result['cv_mean']
                        })
                    else:
                        comparison_data.append({
                            'Task': task_name.replace('_', ' ').title(),
                            'Model': model_name.title(),
                            'R²': result['r2'],
                            'RMSE': result['rmse']
                        })
            
            df = pd.DataFrame(comparison_data)
            
            if 'Accuracy' in df.columns:
                accuracy_df = df.dropna(subset=['Accuracy'])
                if not accuracy_df.empty:
                    fig = px.bar(
                        accuracy_df, 
                        x='Model', 
                        y='Accuracy', 
                        color='Task',
                        title="Classification Model Accuracy",
                        barmode='group'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            if 'R²' in df.columns:
                r2_df = df.dropna(subset=['R²'])
                if not r2_df.empty:
                    fig = px.bar(
                        r2_df, 
                        x='Model', 
                        y='R²', 
                        color='Task',
                        title="Regression Model R² Scores",
                        barmode='group'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Display performance table
            st.subheader("Detailed Performance Metrics")
            st.dataframe(df, use_container_width=True)
            
        else:
            st.warning("No training results found. Please run the model training pipeline first.")
            
    except Exception as e:
        st.error(f"Error loading model performance data: {e}")

elif page == "Historical Analysis":
    st.header("Historical Analysis")
    
    # Load historical data
    try:
        processor = SurvivorDataProcessor()
        df = processor.load_data()
        
        if df is not None:
            st.subheader("Dataset Overview")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Contestants", len(df))
            with col2:
                st.metric("Seasons Covered", f"{df['Season'].min()}-{df['Season'].max()}")
            with col3:
                st.metric("Merge Rate", f"{(df['Made_Merge'] == 'Y').mean():.1%}")
            with col4:
                st.metric("Finale Rate", f"{(df['Made_Finale'] == 'Y').mean():.1%}")
            
            # Success by archetype
            st.subheader("Success Rate by Strategic Archetype")
            archetype_success = df.groupby('Strategic_Archetype').agg({
                'Made_Merge': lambda x: (x == 'Y').mean(),
                'Made_Finale': lambda x: (x == 'Y').mean(),
                'Final_Placement': 'mean'
            }).round(3)
            
            fig = px.bar(
                archetype_success.reset_index(),
                x='Strategic_Archetype',
                y='Made_Merge',
                title="Merge Rate by Strategic Archetype"
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Age analysis
            st.subheader("Success by Age")
            age_bins = pd.cut(df['Age'], bins=[0, 25, 30, 35, 40, 100], labels=['18-25', '26-30', '31-35', '36-40', '40+'])
            age_success = df.groupby(age_bins).agg({
                'Made_Merge': lambda x: (x == 'Y').mean(),
                'Made_Finale': lambda x: (x == 'Y').mean()
            }).round(3)
            
            fig = px.line(
                age_success.reset_index(),
                x='Age',
                y=['Made_Merge', 'Made_Finale'],
                title="Success Rate by Age Group"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.error("Unable to load historical data")
            
    except Exception as e:
        st.error(f"Error in historical analysis: {e}")

# Footer
st.markdown("---")
st.markdown("**Survivor Success Predictor** - Enhanced with Pre-Season Analysis and Cast Rankings")
st.markdown("*Disclaimer: Predictions are for entertainment purposes. Actual Survivor outcomes involve many unmeasurable factors.*")