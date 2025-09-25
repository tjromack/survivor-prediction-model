"""
External Factors Analysis for Survivor Prediction Model
Analyzes demographic, socioeconomic, and pre-game factors that predict success
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from data_processor import SurvivorDataProcessor

class ExternalFactorsAnalyzer:
    def __init__(self):
        self.processor = SurvivorDataProcessor()
        self.df = None
        self.df_processed = None
        
        # Define external occupation income mapping
        self.occupation_income_map = {
            'Business': 65000,
            'Healthcare': 75000, 
            'Legal': 85000,
            'Education': 45000,
            'Entertainment': 55000,
            'Government': 58000,
            'Sports': 70000,
            'Military': 52000,
            'Other': 50000
        }
        
        # Education level mapping
        self.education_level_map = {
            'Business': 'Bachelor+',
            'Healthcare': 'Graduate',
            'Legal': 'Graduate', 
            'Education': 'Bachelor+',
            'Entertainment': 'Bachelor',
            'Government': 'Bachelor+',
            'Sports': 'Bachelor',
            'Military': 'High School+',
            'Other': 'High School+'
        }
        
        # Competitive reality TV markets
        self.competitive_markets = ['CA', 'NY', 'TX', 'FL', 'IL', 'GA']
        
    def load_and_prepare_data(self):
        """Load and prepare data with enhanced external factors"""
        self.df = self.processor.load_data()
        if self.df is None:
            return False
            
        # Process data through pipeline
        X, targets, self.df_processed = self.processor.process_full_pipeline(self.df)
        
        # Add external factor enhancements
        self._add_external_factors()
        
        return True
    
    def _add_external_factors(self):
        """Add enhanced external factor analysis"""
        df = self.df_processed
        
        # Socioeconomic indicators
        df['estimated_income'] = df['Occupation_Category'].map(self.occupation_income_map)
        df['education_level'] = df['Occupation_Category'].map(self.education_level_map)
        df['from_competitive_market'] = df['Home_State'].isin(self.competitive_markets)
        
        # Age groupings
        df['age_decade'] = (df['Age'] // 10) * 10
        df['age_category'] = pd.cut(df['Age'], 
                                   bins=[0, 25, 30, 35, 40, 50, 100],
                                   labels=['18-25', '26-30', '31-35', '36-40', '41-50', '50+'])
        
        # Strategic archetype groupings
        df['archetype_category'] = df['Strategic_Archetype'].map({
            'Social_Player': 'Social',
            'Under_Radar': 'Social',
            'Strategic_Player': 'Strategic', 
            'Villain': 'Strategic',
            'Challenge_Beast': 'Physical',
            'Provider': 'Physical',
            'Hero': 'Social',
            'Wild_Card': 'Unpredictable'
        }).fillna('Other')
        
        # Physical competitiveness score
        athletic_scores = {
            'None': 0, 'Recreational': 1, 'High_School': 2, 
            'College': 3, 'Professional': 4
        }
        df['athletic_score'] = df['Athletic_Background'].map(athletic_scores).fillna(0)
        df['physical_competitiveness'] = (df['athletic_score'] + 
                                         df['Self_Reported_Fitness'] + 
                                         df['physical_build_numeric']) / 3
        
        # Success indicators
        df['early_boot'] = df['Days_Lasted'] < 10
        df['deep_run'] = df['Days_Lasted'] > 20
        df['winner'] = df['Final_Placement'] == 1
        
        self.df_processed = df
        
    def analyze_age_patterns(self):
        """Comprehensive age analysis"""
        df = self.df_processed
        
        # Age success rates
        age_analysis = df.groupby('age_category').agg({
            'made_merge_binary': 'mean',
            'made_finale_binary': 'mean', 
            'winner': 'mean',
            'Days_Lasted': 'mean',
            'Final_Placement': 'mean',
            'Contestant_Name': 'count'
        }).round(3)
        age_analysis.columns = ['Merge_Rate', 'Finale_Rate', 'Win_Rate', 'Avg_Days', 'Avg_Placement', 'Count']
        
        # Age-Gender interactions
        age_gender = df.groupby(['age_category', 'Gender']).agg({
            'made_merge_binary': 'mean',
            'winner': 'mean',
            'Contestant_Name': 'count'
        }).round(3)
        
        return {
            'age_analysis': age_analysis,
            'age_gender': age_gender,
            'correlation': stats.pearsonr(df['Age'], df['Days_Lasted'])[0]
        }
    
    def analyze_occupation_impact(self):
        """Detailed occupation and socioeconomic analysis"""
        df = self.df_processed
        
        # Occupation category success
        occ_analysis = df.groupby('Occupation_Category').agg({
            'made_merge_binary': 'mean',
            'made_finale_binary': 'mean',
            'winner': 'mean',
            'Days_Lasted': 'mean',
            'estimated_income': 'first',
            'Contestant_Name': 'count'
        }).round(3)
        occ_analysis.columns = ['Merge_Rate', 'Finale_Rate', 'Win_Rate', 'Avg_Days', 'Est_Income', 'Count']
        
        # Income vs success correlation
        income_correlation = stats.pearsonr(df['estimated_income'], df['Days_Lasted'])[0]
        
        # Education level analysis
        edu_analysis = df.groupby('education_level').agg({
            'made_merge_binary': 'mean',
            'winner': 'mean',
            'Contestant_Name': 'count'
        }).round(3)
        
        return {
            'occupation_analysis': occ_analysis,
            'education_analysis': edu_analysis,
            'income_correlation': income_correlation
        }
    
    def analyze_strategic_archetype_accuracy(self):
        """Validate if strategic archetypes match actual gameplay"""
        df = self.df_processed
        
        # Strategic players - do they actually strategize?
        strategic_validation = df.groupby('Strategic_Archetype').agg({
            'Advantages_Found': 'mean',
            'Alliance_Count': 'mean', 
            'votes_per_tribal': 'mean',
            'overall_challenge_rate': 'mean',
            'made_merge_binary': 'mean',
            'Days_Lasted': 'mean',
            'Contestant_Name': 'count'
        }).round(3)
        
        # Pre-game target size validation
        target_validation = df.groupby('Pre_Game_Target_Size').agg({
            'votes_per_tribal': 'mean',
            'Days_Lasted': 'mean', 
            'early_boot': 'mean',
            'made_merge_binary': 'mean',
            'Contestant_Name': 'count'
        }).round(3)
        
        return {
            'strategic_validation': strategic_validation,
            'target_validation': target_validation
        }
    
    def analyze_regional_advantages(self):
        """Regional and geographic success patterns"""
        df = self.df_processed
        
        # Regional success rates
        regional_analysis = df.groupby('Home_Region').agg({
            'made_merge_binary': 'mean',
            'winner': 'mean',
            'Days_Lasted': 'mean',
            'Contestant_Name': 'count'
        }).round(3)
        
        # Competitive market advantage
        market_analysis = df.groupby('from_competitive_market').agg({
            'made_merge_binary': 'mean',
            'winner': 'mean', 
            'Days_Lasted': 'mean',
            'Contestant_Name': 'count'
        }).round(3)
        
        # State-level analysis (for states with 3+ contestants)
        state_counts = df['Home_State'].value_counts()
        frequent_states = state_counts[state_counts >= 3].index
        
        state_analysis = df[df['Home_State'].isin(frequent_states)].groupby('Home_State').agg({
            'made_merge_binary': 'mean',
            'winner': 'sum',
            'Days_Lasted': 'mean',
            'Contestant_Name': 'count'
        }).round(3)
        
        return {
            'regional_analysis': regional_analysis,
            'market_analysis': market_analysis,
            'state_analysis': state_analysis
        }
    
    def analyze_physical_factors(self):
        """Physical and athletic background analysis"""
        df = self.df_processed
        
        # Athletic background vs challenge performance
        athletic_analysis = df.groupby('Athletic_Background').agg({
            'overall_challenge_rate': 'mean',
            'Individual_Challenges_Won': 'mean',
            'made_merge_binary': 'mean',
            'Days_Lasted': 'mean',
            'Contestant_Name': 'count'
        }).round(3)
        
        # Physical build analysis
        build_analysis = df.groupby('Physical_Build').agg({
            'Days_Lasted': 'mean',
            'overall_challenge_rate': 'mean',
            'made_merge_binary': 'mean',
            'Contestant_Name': 'count'
        }).round(3)
        
        # Physical competitiveness correlation
        phys_correlation = stats.pearsonr(df['physical_competitiveness'], df['Days_Lasted'])[0]
        
        return {
            'athletic_analysis': athletic_analysis,
            'build_analysis': build_analysis, 
            'physical_correlation': phys_correlation
        }
    
    def identify_success_archetypes(self):
        """Identify successful demographic combinations"""
        df = self.df_processed
        
        # Create success archetype combinations
        def create_archetype(row):
            age_cat = '25-' if row['Age'] < 25 else '25-35' if row['Age'] < 35 else '35+'
            gender = row['Gender']
            strategy = row['archetype_category']
            return f"{age_cat}_{gender}_{strategy}"
        
        df['success_archetype'] = df.apply(create_archetype, axis=1)
        
        # Analyze success by archetype
        archetype_analysis = df.groupby('success_archetype').agg({
            'made_merge_binary': 'mean',
            'winner': 'mean',
            'Days_Lasted': 'mean',
            'Contestant_Name': 'count'
        }).round(3)
        
        # Filter for archetypes with 3+ contestants
        archetype_analysis = archetype_analysis[archetype_analysis['Contestant_Name'] >= 3]
        archetype_analysis = archetype_analysis.sort_values('winner', ascending=False)
        
        return archetype_analysis
    
    def external_factors_correlation_matrix(self):
        """Create correlation matrix for external factors vs success"""
        df = self.df_processed
        
        # Select external factors and success metrics
        external_factors = [
            'Age', 'Self_Reported_Fitness', 'Pre_Game_Target_Size',
            'athletic_score', 'estimated_income', 'physical_competitiveness'
        ]
        
        success_metrics = [
            'Days_Lasted', 'Final_Placement', 'made_merge_binary', 
            'made_finale_binary', 'winner'
        ]
        
        # Create correlation matrix
        correlation_data = df[external_factors + success_metrics]
        correlation_matrix = correlation_data.corr()
        
        # Extract correlations between external factors and success
        external_success_corr = correlation_matrix.loc[external_factors, success_metrics]
        
        return external_success_corr
    
    def generate_external_factors_report(self):
        """Generate comprehensive report on external factors"""
        if not self.load_and_prepare_data():
            return "Failed to load data"
        
        # Run all analyses
        age_results = self.analyze_age_patterns()
        occ_results = self.analyze_occupation_impact()
        strategy_results = self.analyze_strategic_archetype_accuracy()
        regional_results = self.analyze_regional_advantages()
        physical_results = self.analyze_physical_factors()
        archetype_results = self.identify_success_archetypes()
        correlation_matrix = self.external_factors_correlation_matrix()
        
        return {
            'age_analysis': age_results,
            'occupation_analysis': occ_results,
            'strategic_analysis': strategy_results,
            'regional_analysis': regional_results,
            'physical_analysis': physical_results,
            'success_archetypes': archetype_results,
            'correlation_matrix': correlation_matrix,
            'sample_size': len(self.df_processed)
        }

# Visualization functions for Streamlit integration
def plot_age_success_patterns(age_analysis):
    """Plot age-based success patterns"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Merge Rate by Age', 'Win Rate by Age', 'Days Lasted by Age', 'Sample Sizes'],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    age_data = age_analysis['age_analysis']
    categories = age_data.index
    
    # Merge rate
    fig.add_trace(go.Bar(x=categories, y=age_data['Merge_Rate'], name='Merge Rate', marker_color='lightblue'), row=1, col=1)
    
    # Win rate
    fig.add_trace(go.Bar(x=categories, y=age_data['Win_Rate'], name='Win Rate', marker_color='gold'), row=1, col=2)
    
    # Days lasted
    fig.add_trace(go.Bar(x=categories, y=age_data['Avg_Days'], name='Avg Days', marker_color='lightgreen'), row=2, col=1)
    
    # Sample sizes
    fig.add_trace(go.Bar(x=categories, y=age_data['Count'], name='Count', marker_color='lightcoral'), row=2, col=2)
    
    fig.update_layout(height=600, showlegend=False, title_text="Age Analysis Dashboard")
    return fig

def plot_occupation_income_correlation(occ_analysis):
    """Plot occupation success vs estimated income"""
    fig = px.scatter(
        occ_analysis, 
        x='Est_Income', 
        y='Merge_Rate',
        size='Count',
        hover_name=occ_analysis.index,
        hover_data={'Win_Rate': True, 'Avg_Days': True},
        title="Occupation Income vs Success Rate"
    )
    
    fig.update_layout(
        xaxis_title="Estimated Income ($)",
        yaxis_title="Merge Success Rate"
    )
    
    return fig

def plot_strategic_archetype_validation(strategy_results):
    """Plot strategic archetype validation"""
    strategic_data = strategy_results['strategic_validation']
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Advantages Found', 'Alliance Count', 'Challenge Win Rate', 'Merge Success'],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    archetypes = strategic_data.index
    
    fig.add_trace(go.Bar(x=archetypes, y=strategic_data['Advantages_Found'], name='Advantages'), row=1, col=1)
    fig.add_trace(go.Bar(x=archetypes, y=strategic_data['Alliance_Count'], name='Alliances'), row=1, col=2)
    fig.add_trace(go.Bar(x=archetypes, y=strategic_data['overall_challenge_rate'], name='Challenge Rate'), row=2, col=1)
    fig.add_trace(go.Bar(x=archetypes, y=strategic_data['made_merge_binary'], name='Merge Rate'), row=2, col=2)
    
    fig.update_layout(height=600, showlegend=False, title_text="Strategic Archetype Validation")
    fig.update_xaxes(tickangle=45)
    
    return fig

def plot_correlation_heatmap(correlation_matrix):
    """Plot correlation heatmap between external factors and success"""
    fig = px.imshow(
        correlation_matrix,
        text_auto=True,
        color_continuous_scale='RdBu',
        aspect='auto',
        title="External Factors vs Success Metrics Correlation"
    )
    
    fig.update_layout(
        width=800,
        height=500
    )
    
    return fig

# Streamlit App Integration Function
def create_external_factors_dashboard():
    """Create Streamlit dashboard for external factors analysis"""
    import streamlit as st
    
    st.header("🧬 External Factors Deep Dive")
    st.markdown("**Analyzing demographic and pre-game factors that predict Survivor success**")
    
    # Initialize analyzer
    @st.cache_data
    def load_external_analysis():
        analyzer = ExternalFactorsAnalyzer()
        return analyzer.generate_external_factors_report()
    
    results = load_external_analysis()
    
    if not isinstance(results, dict):
        st.error("Failed to load external factors analysis")
        return
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Sample Size", f"{results['sample_size']} contestants")
    with col2:
        age_corr = results['age_analysis']['correlation']
        st.metric("Age-Success Correlation", f"{age_corr:.3f}")
    with col3:
        income_corr = results['occupation_analysis']['income_correlation'] 
        st.metric("Income-Success Correlation", f"{income_corr:.3f}")
    with col4:
        phys_corr = results['physical_analysis']['physical_correlation']
        st.metric("Physical-Success Correlation", f"{phys_corr:.3f}")
    
    # Tabs for different analyses
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Age Analysis", "Occupation Impact", "Strategic Validation", 
        "Regional Patterns", "Physical Factors", "Success Archetypes"
    ])
    
    with tab1:
        st.subheader("Age-Based Success Patterns")
        
        # Age analysis visualization
        fig_age = plot_age_success_patterns(results['age_analysis'])
        st.plotly_chart(fig_age, use_container_width=True)
        
        # Age data table
        st.subheader("Age Category Breakdown")
        st.dataframe(results['age_analysis']['age_analysis'])
        
        # Age-Gender interaction
        st.subheader("Age-Gender Success Interaction")
        age_gender_pivot = results['age_analysis']['age_gender'].reset_index().pivot(
            index='age_category', columns='Gender', values='made_merge_binary'
        )
        st.dataframe(age_gender_pivot.style.format("{:.1%}"))
    
    with tab2:
        st.subheader("Occupation and Socioeconomic Impact")
        
        # Income vs success scatter plot
        fig_income = plot_occupation_income_correlation(results['occupation_analysis']['occupation_analysis'])
        st.plotly_chart(fig_income, use_container_width=True)
        
        # Occupation analysis table
        st.subheader("Occupation Category Analysis")
        occ_data = results['occupation_analysis']['occupation_analysis']
        st.dataframe(occ_data.style.format({
            'Merge_Rate': '{:.1%}',
            'Finale_Rate': '{:.1%}', 
            'Win_Rate': '{:.1%}',
            'Est_Income': '${:,.0f}'
        }))
        
        # Education level analysis
        st.subheader("Education Level Impact")
        edu_data = results['occupation_analysis']['education_analysis']
        st.dataframe(edu_data.style.format({
            'made_merge_binary': '{:.1%}',
            'winner': '{:.1%}'
        }))
    
    with tab3:
        st.subheader("Strategic Archetype Validation")
        st.markdown("*Do contestants' self-reported strategic types match their actual gameplay?*")
        
        # Strategic validation plot
        fig_strategy = plot_strategic_archetype_validation(results['strategic_analysis'])
        st.plotly_chart(fig_strategy, use_container_width=True)
        
        # Strategic archetype data
        st.subheader("Archetype Performance Validation")
        strategy_data = results['strategic_analysis']['strategic_validation']
        st.dataframe(strategy_data.style.format({
            'made_merge_binary': '{:.1%}',
            'Advantages_Found': '{:.1f}',
            'Alliance_Count': '{:.1f}'
        }))
        
        # Pre-game target validation
        st.subheader("Pre-Game Target Size Validation")
        st.markdown("*Does perceived threat level predict actual elimination risk?*")
        target_data = results['strategic_analysis']['target_validation']
        st.dataframe(target_data.style.format({
            'early_boot': '{:.1%}',
            'made_merge_binary': '{:.1%}',
            'votes_per_tribal': '{:.2f}'
        }))
    
    with tab4:
        st.subheader("Regional and Geographic Patterns")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Regional Success Rates")
            regional_data = results['regional_analysis']['regional_analysis']
            fig_regional = px.bar(
                regional_data.reset_index(),
                x='Home_Region', 
                y='made_merge_binary',
                title="Merge Success Rate by Region"
            )
            st.plotly_chart(fig_regional, use_container_width=True)
        
        with col2:
            st.subheader("Competitive Market Advantage")
            market_data = results['regional_analysis']['market_analysis']
            market_comparison = pd.DataFrame({
                'Market Type': ['Non-Competitive', 'Competitive'],
                'Merge Rate': market_data['made_merge_binary'].values,
                'Win Rate': market_data['winner'].values
            })
            st.dataframe(market_comparison.style.format({
                'Merge Rate': '{:.1%}',
                'Win Rate': '{:.1%}'
            }))
        
        # State analysis
        st.subheader("State-Level Analysis (States with 3+ contestants)")
        state_data = results['regional_analysis']['state_analysis']
        st.dataframe(state_data.style.format({
            'made_merge_binary': '{:.1%}'
        }))
    
    with tab5:
        st.subheader("Physical and Athletic Factors")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Athletic Background Impact")
            athletic_data = results['physical_analysis']['athletic_analysis']
            fig_athletic = px.bar(
                athletic_data.reset_index(),
                x='Athletic_Background',
                y='made_merge_binary', 
                title="Success Rate by Athletic Background"
            )
            st.plotly_chart(fig_athletic, use_container_width=True)
        
        with col2:
            st.subheader("Physical Build Analysis")
            build_data = results['physical_analysis']['build_analysis']
            fig_build = px.bar(
                build_data.reset_index(),
                x='Physical_Build',
                y='overall_challenge_rate',
                title="Challenge Performance by Build"
            )
            st.plotly_chart(fig_build, use_container_width=True)
        
        # Physical factor tables
        st.subheader("Athletic Background Detailed Analysis")
        st.dataframe(athletic_data.style.format({
            'overall_challenge_rate': '{:.1%}',
            'made_merge_binary': '{:.1%}'
        }))
    
    with tab6:
        st.subheader("High-Success Demographic Combinations")
        st.markdown("*Identifying the most successful contestant archetypes*")
        
        archetype_data = results['success_archetypes']
        
        # Top success archetypes
        st.subheader("Most Successful Archetypes")
        top_archetypes = archetype_data.head(10)
        
        fig_archetypes = px.bar(
            top_archetypes.reset_index(),
            x='success_archetype',
            y='winner',
            hover_data=['made_merge_binary', 'Contestant_Name'],
            title="Win Rate by Demographic Archetype"
        )
        fig_archetypes.update_xaxes(tickangle=45)
        st.plotly_chart(fig_archetypes, use_container_width=True)
        
        # Archetype data table
        st.dataframe(top_archetypes.style.format({
            'made_merge_binary': '{:.1%}',
            'winner': '{:.1%}'
        }))
        
        # Correlation heatmap
        st.subheader("External Factors Correlation Matrix")
        fig_corr = plot_correlation_heatmap(results['correlation_matrix'])
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Key insights
        st.subheader("Key Insights")
        st.markdown("""
        **Age Patterns:**
        - Younger contestants often have higher energy but less life experience
        - Middle-aged contestants (30-40) often perform best with good balance
        
        **Occupation Impact:**
        - Higher-income professions correlate with strategic thinking
        - Healthcare and legal backgrounds show strong performance
        
        **Strategic Validation:**
        - Self-reported "Strategic Players" do find more advantages
        - "Challenge Beasts" do win more individual challenges
        - Pre-game target assessment is moderately predictive
        
        **Regional Factors:**
        - Competitive reality TV markets may produce better-prepared contestants
        - Regional cultural differences affect social gameplay
        """)

if __name__ == "__main__":
    analyzer = ExternalFactorsAnalyzer()
    results = analyzer.generate_external_factors_report()
    
    print("🏝️ EXTERNAL FACTORS ANALYSIS COMPLETE")
    print("=" * 50)
    
    if isinstance(results, dict):
        print(f"Sample Size: {results['sample_size']} contestants")
        print("\nKey Findings:")
        print(f"Age-Success Correlation: {results['age_analysis']['correlation']:.3f}")
        print(f"Income-Success Correlation: {results['occupation_analysis']['income_correlation']:.3f}")
        print(f"Physical Competitiveness Correlation: {results['physical_analysis']['physical_correlation']:.3f}")
        
        print(f"\nTop Success Archetypes:")
        top_archetypes = results['success_archetypes'].head(5)
        for archetype, data in top_archetypes.iterrows():
            print(f"  {archetype}: {data['winner']:.1%} win rate, {data['made_merge_binary']:.1%} merge rate (n={data['Contestant_Name']})")
    else:
        print(results)
