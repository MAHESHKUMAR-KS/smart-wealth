"""
Manual Training Script for AI Risk Assessment Models
Use this script to manually train the AI models with your own data
"""
import pandas as pd
import numpy as np
from ai_risk_assessment import UserBehaviorAnalyzer, UserClusterAnalyzer, EnhancedRiskProfiler
import json


def create_sample_training_data():
    """
    Create sample training data for demonstration
    In practice, you would load this from a CSV file or database
    """
    # Sample user profiles for training
    sample_profiles = [
        {
            'age': 25, 'investment_amount': 30000, 'horizon': 15,
            'risk_reaction': 'Buy more', 'goal': 'wealth_creation',
            'risk_profile': 'aggressive'
        },
        {
            'age': 35, 'investment_amount': 75000, 'horizon': 10,
            'risk_reaction': 'Hold steady', 'goal': 'retirement',
            'risk_profile': 'moderate'
        },
        {
            'age': 45, 'investment_amount': 150000, 'horizon': 8,
            'risk_reaction': 'Feel concerned', 'goal': 'child_education',
            'risk_profile': 'moderate'
        },
        {
            'age': 55, 'investment_amount': 200000, 'horizon': 5,
            'risk_reaction': 'Panic and sell', 'goal': 'retirement',
            'risk_profile': 'conservative'
        },
        {
            'age': 30, 'investment_amount': 50000, 'horizon': 12,
            'risk_reaction': 'Hold steady', 'goal': 'home_purchase',
            'risk_profile': 'moderate'
        },
        {
            'age': 28, 'investment_amount': 40000, 'horizon': 20,
            'risk_reaction': 'Buy more', 'goal': 'wealth_creation',
            'risk_profile': 'aggressive'
        },
        {
            'age': 40, 'investment_amount': 100000, 'horizon': 7,
            'risk_reaction': 'Feel concerned', 'goal': 'child_education',
            'risk_profile': 'moderate'
        },
        {
            'age': 50, 'investment_amount': 180000, 'horizon': 6,
            'risk_reaction': 'Hold steady', 'goal': 'retirement',
            'risk_profile': 'conservative'
        },
        {
            'age': 32, 'investment_amount': 60000, 'horizon': 11,
            'risk_reaction': 'Buy more', 'goal': 'wealth_creation',
            'risk_profile': 'aggressive'
        },
        {
            'age': 38, 'investment_amount': 85000, 'horizon': 9,
            'risk_reaction': 'Hold steady', 'goal': 'retirement',
            'risk_profile': 'moderate'
        },
        {
            'age': 42, 'investment_amount': 120000, 'horizon': 8,
            'risk_reaction': 'Feel concerned', 'goal': 'child_education',
            'risk_profile': 'moderate'
        },
        {
            'age': 48, 'investment_amount': 160000, 'horizon': 5,
            'risk_reaction': 'Panic and sell', 'goal': 'retirement',
            'risk_profile': 'conservative'
        },
        {
            'age': 27, 'investment_amount': 35000, 'horizon': 18,
            'risk_reaction': 'Buy more', 'goal': 'wealth_creation',
            'risk_profile': 'aggressive'
        },
        {
            'age': 33, 'investment_amount': 65000, 'horizon': 12,
            'risk_reaction': 'Hold steady', 'goal': 'home_purchase',
            'risk_profile': 'moderate'
        },
        {
            'age': 37, 'investment_amount': 95000, 'horizon': 10,
            'risk_reaction': 'Feel concerned', 'goal': 'child_education',
            'risk_profile': 'moderate'
        }
    ]
    
    return sample_profiles


def load_training_data_from_csv(csv_file_path):
    """
    Load training data from a CSV file
    Expected columns: age, investment_amount, horizon, risk_reaction, goal, risk_profile
    """
    try:
        df = pd.read_csv(csv_file_path)
        # Convert DataFrame to list of dictionaries
        profiles = df.to_dict('records')
        return profiles
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None


def manual_train_behavior_model(training_data):
    """
    Manually train the behavior analysis model
    """
    print("Training User Behavior Analysis Model...")
    
    # Initialize the behavior analyzer
    behavior_analyzer = UserBehaviorAnalyzer()
    
    try:
        # Train the model
        results = behavior_analyzer.train_behavior_model(training_data)
        
        print("Behavior Model Training Completed!")
        print(f"Accuracy: {results['accuracy']:.2f}")
        print("\nFeature Importance:")
        for feature, importance in results['feature_importance'].items():
            print(f"  {feature}: {importance:.3f}")
        
        # Save the trained model
        behavior_analyzer.save_model("behavior_model.pkl")
        print("\nBehavior model saved as 'behavior_model.pkl'")
        
        return behavior_analyzer, results
    except Exception as e:
        print(f"Error training behavior model: {e}")
        return None, None


def manual_train_clustering_model(training_data):
    """
    Manually train the clustering model
    """
    print("\nTraining User Clustering Model...")
    
    # Initialize the cluster analyzer
    cluster_analyzer = UserClusterAnalyzer(n_clusters=5)
    
    try:
        # Fit the clustering model
        results = cluster_analyzer.fit_clusters(training_data)
        
        print("Clustering Model Training Completed!")
        print(f"Inertia: {results['inertia']:.2f}")
        print(f"Number of clusters: {results['n_clusters']}")
        
        print("\nCluster Statistics:")
        for cluster_name, stats in results['cluster_stats'].items():
            print(f"  {cluster_name}:")
            print(f"    Size: {stats['size']}")
            print(f"    Avg Age: {stats['avg_age']:.1f}")
            print(f"    Avg Investment: ₹{stats['avg_investment']:,.0f}")
            print(f"    Avg Horizon: {stats['avg_horizon']:.1f} years")
        
        # Save the trained model
        cluster_analyzer.save_model("cluster_model.pkl")
        print("\nClustering model saved as 'cluster_model.pkl'")
        
        return cluster_analyzer, results
    except Exception as e:
        print(f"Error training clustering model: {e}")
        return None, None


def test_trained_models():
    """
    Test the trained models with a new user profile
    """
    print("\nTesting Trained Models with New User Profile...")
    
    # Create enhanced risk profiler
    profiler = EnhancedRiskProfiler()
    
    # Load trained models
    try:
        profiler.load_models(".")
        print("Trained models loaded successfully!")
    except Exception as e:
        print(f"Error loading models: {e}")
        return
    
    # Test with a new user
    new_user = {
        'age': 29, 'investment_amount': 45000, 'horizon': 13,
        'risk_reaction': 'Buy more', 'goal': 'wealth_creation',
        'risk_profile': 'aggressive'  # Traditional profile
    }
    
    # Get enhanced risk profile
    enhanced_profile = profiler.enhanced_risk_profile(new_user)
    
    print("\nEnhanced Risk Profile Results:")
    print(f"Traditional Profile: {enhanced_profile['traditional_profile']}")
    print(f"AI Predicted Profile: {enhanced_profile['ai_predicted_profile']}")
    print(f"Prediction Confidence: {enhanced_profile['prediction_confidence']:.2f}")
    print(f"Cluster Assignment: {enhanced_profile['cluster_assignment']}")
    print(f"Similar Users Count: {enhanced_profile['similar_users_count']}")
    print(f"Final Profile: {enhanced_profile['final_profile']}")


def main():
    """
    Main function to demonstrate manual training
    """
    print("AI Risk Assessment Manual Training Script")
    print("=" * 50)
    
    # Option 1: Use sample data
    print("\nOption 1: Using sample training data")
    training_data = create_sample_training_data()
    print(f"Loaded {len(training_data)} sample profiles")
    
    # Train behavior model
    behavior_model, behavior_results = manual_train_behavior_model(training_data)
    
    # Train clustering model
    cluster_model, cluster_results = manual_train_clustering_model(training_data)
    
    # Test trained models
    if behavior_model and cluster_model:
        test_trained_models()
    
    print("\n" + "=" * 50)
    print("Training completed!")
    
    # Option 2: Load from CSV (commented out for demonstration)
    # print("\nOption 2: Loading from CSV file")
    # csv_data = load_training_data_from_csv("your_training_data.csv")
    # if csv_data:
    #     print(f"Loaded {len(csv_data)} profiles from CSV")
    #     manual_train_behavior_model(csv_data)
    #     manual_train_clustering_model(csv_data)


if __name__ == "__main__":
    main()