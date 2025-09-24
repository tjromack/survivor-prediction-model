"""
Advanced model evaluation and analysis for Survivor prediction models
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report, roc_curve, auc, 
                           precision_recall_curve, average_precision_score)
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class SurvivorModelEvaluator:
    def __init__(self):
        self.results = None
        self.load_results()
    
    def load_results(self):
        """Load training results"""
        try:
            results_file = Path(__file__).parent.parent / 'models' / 'training_results.pkl'
            if results_file.exists():
                self.results = joblib.load(results_file)
                print("✅ Training results loaded successfully")
            else:
                print("❌ No training results found. Run model_trainer.py first!")
        except Exception as e:
            print(f"Error loading results: {e}")
    
    def plot_confusion_matrices(self):
        """Plot confusion matrices for all classification models"""
        if not self.results:
            return
        
        classification_tasks = ['merge_prediction', 'finale_prediction', 'winner_prediction']
        
        for task in classification_tasks:
            if task not in self.results:
                continue
                
            task_results = self.results[task]
            n_models = len(task_results)
            
            fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
            if n_models == 1:
                axes = [axes]
            
            for i, (model_name, result) in enumerate(task_results.items()):
                if 'confusion_matrix' in result:
                    cm = result['confusion_matrix']
                    
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                              ax=axes[i], cbar_kws={'shrink': 0.8})
                    axes[i].set_title(f'{model_name.title()} - {task.replace("_", " ").title()}')
                    axes[i].set_xlabel('Predicted')
                    axes[i].set_ylabel('Actual')
            
            plt.tight_layout()
            plt.show()
    
    def plot_roc_curves(self):
        """Plot ROC curves for classification models"""
        if not self.results:
            return
        
        classification_tasks = ['merge_prediction', 'finale_prediction', 'winner_prediction']
        
        for task in classification_tasks:
            if task not in self.results:
                continue
                
            plt.figure(figsize=(8, 6))
            
            for model_name, result in self.results[task].items():
                if 'y_test' in result and 'y_pred_proba' in result and result['y_pred_proba'] is not None:
                    y_test = result['y_test']
                    y_proba = result['y_pred_proba']
                    
                    fpr, tpr, _ = roc_curve(y_test, y_proba)
                    roc_auc = auc(fpr, tpr)
                    
                    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
            
            plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curves - {task.replace("_", " ").title()}')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            plt.show()
    
    def plot_prediction_vs_actual(self):
        """Plot prediction vs actual for regression models"""
        if not self.results:
            return
        
        regression_tasks = ['placement_prediction', 'days_lasted_prediction']
        
        for task in regression_tasks:
            if task not in self.results:
                continue
                
            task_results = self.results[task]
            n_models = len(task_results)
            
            fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
            if n_models == 1:
                axes = [axes]
            
            for i, (model_name, result) in enumerate(task_results.items()):
                if 'y_test' in result and 'y_pred' in result:
                    y_test = result['y_test']
                    y_pred = result['y_pred']
                    
                    axes[i].scatter(y_test, y_pred, alpha=0.6, color='steelblue')
                    
                    # Perfect prediction line
                    min_val = min(y_test.min(), y_pred.min())
                    max_val = max(y_test.max(), y_pred.max())
                    axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
                    
                    axes[i].set_xlabel('Actual')
                    axes[i].set_ylabel('Predicted')
                    axes[i].set_title(f'{model_name.title()} - {task.replace("_", " ").title()}')
                    axes[i].grid(True, alpha=0.3)
                    
                    # Add R² score
                    if 'r2' in result:
                        axes[i].text(0.05, 0.95, f'R² = {result["r2"]:.3f}', 
                                   transform=axes[i].transAxes, fontsize=12,
                                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            plt.show()
    
    def create_performance_summary(self):
        """Create a comprehensive performance summary"""
        if not self.results:
            return
        
        summary_data = []
        
        for task_name, task_results in self.results.items():
            for model_name, result in task_results.items():
                row = {
                    'Task': task_name.replace('_', ' ').title(),
                    'Model': model_name.title(),
                }
                
                if 'accuracy' in result:  # Classification
                    row.update({
                        'Accuracy': f"{result['accuracy']:.3f}",
                        'CV Score': f"{result['cv_mean']:.3f} (±{result['cv_std']:.3f})",
                        'Type': 'Classification'
                    })
                else:  # Regression
                    row.update({
                        'R²': f"{result['r2']:.3f}",
                        'RMSE': f"{result['rmse']:.3f}",
                        'CV RMSE': f"{result['cv_rmse_mean']:.3f} (±{result['cv_rmse_std']:.3f})",
                        'Type': 'Regression'
                    })
                
                summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        return summary_df
    
    def analyze_feature_importance_patterns(self):
        """Analyze patterns in feature importance across models"""
        if not self.results:
            return
        
        # Collect all feature importances
        all_importances = {}
        
        for task_name, task_results in self.results.items():
            for model_name, result in task_results.items():
                if 'feature_importance' in result:
                    key = f"{task_name}_{model_name}"
                    all_importances[key] = result['feature_importance'].set_index('feature')['importance']
        
        if not all_importances:
            print("No feature importance data available")
            return
        
        # Combine into a single DataFrame
        importance_df = pd.DataFrame(all_importances).fillna(0)
        
        # Calculate mean importance across all models
        importance_df['mean_importance'] = importance_df.mean(axis=1)
        importance_df = importance_df.sort_values('mean_importance', ascending=False)
        
        # Plot top features
        top_features = importance_df.head(20)
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(top_features)), top_features['mean_importance'], 
                color='steelblue', alpha=0.7)
        plt.yticks(range(len(top_features)), top_features.index)
        plt.xlabel('Mean Feature Importance')
        plt.title('Top 20 Most Important Features Across All Models')
        plt.gca().invert_yaxis()
        
        # Add value labels
        for i, importance in enumerate(top_features['mean_importance']):
            plt.text(importance + 0.001, i, f'{importance:.3f}', 
                    va='center', ha='left')
        
        plt.tight_layout()
        plt.show()
        
        return importance_df

if __name__ == "__main__":
    evaluator = SurvivorModelEvaluator()
    
    if evaluator.results:
        print("📊 SURVIVOR MODEL EVALUATION")
        print("=" * 50)
        
        # Performance summary
        summary = evaluator.create_performance_summary()
        print("\n🎯 Performance Summary:")
        print(summary.to_string(index=False))
        
        # Plot evaluations
        print("\n📈 Creating visualizations...")
        evaluator.plot_confusion_matrices()
        evaluator.plot_roc_curves()
        evaluator.plot_prediction_vs_actual()
        
        # Feature importance analysis
        print("\n🔍 Analyzing feature importance patterns...")
        importance_df = evaluator.analyze_feature_importance_patterns()
        
        if importance_df is not None:
            print("\nTop 10 Most Important Features Overall:")
            print(importance_df[['mean_importance']].head(10).to_string())