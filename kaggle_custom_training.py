"""
Kaggle Custom Dataset Training Script
Use this script to train AI models with your own dataset on Kaggle
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os

def load_custom_dataset(file_path):
    """
    Load your custom dataset
    Expected columns:
    - age: integer
    - investment_amount: float
    - horizon: integer (years)
    - risk_reaction: string (options: "Panic and sell", "Feel concerned", "Hold steady", "Buy more")
    - goal: string (options: "emergency_fund", "short_term_savings", "home_purchase", "child_education", "retirement", "wealth_creation")
    - risk_profile: string (options: "conservative", "moderate", "aggressive")
    """
    try:
        # Load dataset
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully with {len(df)} rows")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def preprocess_data(df):
    """
    Preprocess the data for training
    """
    # Create risk reaction scores
    risk_reaction_map = {
        "Panic and sell": 1,
        "Feel concerned": 2,
        "Hold steady": 3,
        "Buy more": 4
    }
    df['risk_reaction_score'] = df['risk_reaction'].map(risk_reaction_map)
    
    # Create return preference scores based on goals
    goal_return_map = {
        'emergency_fund': 1,
        'short_term_savings': 1.5,
        'home_purchase': 2,
        'child_education': 2.5,
        'retirement': 3,
        'wealth_creation': 3.5
    }
    df['return_preference_score'] = df['goal'].map(goal_return_map)
    
    # Estimate market knowledge based on investment amount
    df['market_knowledge_score'] = np.log(df['investment_amount'] + 1) / 10
    
    # Estimate investment frequency based on horizon
    df['investment_frequency'] = np.clip(df['horizon'] / 3, 1, 5)
    
    return df

class CustomUserBehaviorAnalyzer:
    """Train behavior analysis model with custom dataset"""
    
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def prepare_features(self, df):
        """Prepare features for training"""
        features = df[['age', 'investment_amount', 'horizon', 
                      'risk_reaction_score', 'return_preference_score',
                      'market_knowledge_score', 'investment_frequency']]
        return features
    
    def train(self, df):
        """Train the model"""
        # Prepare data
        X = self.prepare_features(df)
        y = df['risk_profile']
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred),
            'feature_importance': dict(zip(['age', 'investment_amount', 'horizon', 
                                          'risk_reaction_score', 'return_preference_score',
                                          'market_knowledge_score', 'investment_frequency'],
                                         self.model.feature_importances_))
        }
    
    def save_model(self, filepath):
        """Save trained model"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler
            }, f)
        print(f"Model saved to {filepath}")

class CustomUserClusterAnalyzer:
    """Train clustering model with custom dataset"""
    
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=n_clusters, random_state=42)
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def prepare_features(self, df):
        """Prepare features for clustering"""
        # Convert risk profile to numerical score
        risk_map = {'conservative': 1, 'moderate': 2, 'aggressive': 3}
        df['risk_score'] = df['risk_profile'].map(risk_map)
        
        # Calculate return expectation
        df['return_expectation'] = df['risk_score'] * 3 + (df['horizon'] * 0.5)
        df['return_expectation'] = np.clip(df['return_expectation'], 5, 20)
        
        features = df[['age', 'investment_amount', 'horizon', 
                      'risk_score', 'return_expectation']]
        return features
    
    def fit(self, df):
        """Fit clustering model"""
        # Prepare features
        X = self.prepare_features(df)
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit model
        cluster_labels = self.model.fit_predict(X_scaled)
        self.is_fitted = True
        
        # Calculate basic statistics
        df['cluster'] = cluster_labels
        cluster_stats = {}
        for i in range(self.n_clusters):
            cluster_data = df[df['cluster'] == i]
            cluster_stats[f'cluster_{i}'] = {
                'size': len(cluster_data),
                'avg_age': cluster_data['age'].mean(),
                'avg_investment': cluster_data['investment_amount'].mean()
            }
        
        return {
            'cluster_stats': cluster_stats,
            'inertia': self.model.inertia_
        }
    
    def save_model(self, filepath):
        """Save fitted model"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'n_clusters': self.n_clusters
            }, f)
        print(f"Model saved to {filepath}")

def main():
    """
    Main training function - customize this with your dataset path
    """
    print("Kaggle Custom Dataset Training")
    print("=" * 40)
    
    # TODO: Change this to your actual dataset file name
    DATASET_FILE = "your_dataset.csv"  # <-- CHANGE THIS
    
    # Check if dataset file exists
    if not os.path.exists(DATASET_FILE):
        print(f"Dataset file '{DATASET_FILE}' not found!")
        print("Please upload your dataset to Kaggle and update the file name above.")
        print("\nExpected dataset format:")
        print("Columns: age, investment_amount, horizon, risk_reaction, goal, risk_profile")
        print("\nExample rows:")
        print("age,investment_amount,horizon,risk_reaction,goal,risk_profile")
        print("25,50000,15,Buy more,wealth_creation,aggressive")
        print("45,150000,8,Feel concerned,child_education,moderate")
        print("55,200000,5,Panic and sell,retirement,conservative")
        return
    
    # Load and preprocess data
    print(f"Loading dataset: {DATASET_FILE}")
    df = load_custom_dataset(DATASET_FILE)
    
    if df is None:
        return
    
    print("Preprocessing data...")
    df = preprocess_data(df)
    
    # Train behavior analyzer
    print("\nTraining User Behavior Analyzer...")
    behavior_analyzer = CustomUserBehaviorAnalyzer()
    behavior_results = behavior_analyzer.train(df)
    
    print(f"Accuracy: {behavior_results['accuracy']:.3f}")
    print("\nFeature Importance:")
    for feature, importance in behavior_results['feature_importance'].items():
        print(f"  {feature}: {importance:.3f}")
    
    # Train cluster analyzer
    print("\nTraining User Cluster Analyzer...")
    cluster_analyzer = CustomUserClusterAnalyzer(n_clusters=5)
    cluster_results = cluster_analyzer.fit(df)
    
    print(f"Inertia: {cluster_results['inertia']:.2f}")
    print("\nCluster Statistics:")
    for cluster_name, stats in cluster_results['cluster_stats'].items():
        print(f"  {cluster_name}: {stats['size']} users, avg age {stats['avg_age']:.1f}")
    
    # Save models
    print("\nSaving trained models...")
    behavior_analyzer.save_model('behavior_model.pkl')
    cluster_analyzer.save_model('cluster_model.pkl')
    
    print("\nTraining completed successfully!")
    print("Download the .pkl files and use them in your WealthyWise application.")

# For Kaggle notebook usage
def kaggle_training_example():
    """
    Example of how to use this in a Kaggle notebook
    """
    # This is just for demonstration - you would replace this with your actual data loading
    print("Creating sample dataset for demonstration...")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'age': np.random.randint(20, 70, n_samples),
        'investment_amount': np.random.lognormal(10, 1, n_samples),
        'horizon': np.random.randint(1, 30, n_samples),
        'risk_reaction': np.random.choice(['Panic and sell', 'Feel concerned', 'Hold steady', 'Buy more'], n_samples),
        'goal': np.random.choice(['emergency_fund', 'short_term_savings', 'home_purchase', 
                                 'child_education', 'retirement', 'wealth_creation'], n_samples),
        'risk_profile': np.random.choice(['conservative', 'moderate', 'aggressive'], n_samples)
    }
    
    df = pd.DataFrame(data)
    df.to_csv('sample_dataset.csv', index=False)
    print(f"Sample dataset created with {len(df)} rows")
    
    # Now train with this sample data
    global DATASET_FILE
    DATASET_FILE = 'sample_dataset.csv'
    main()

if __name__ == "__main__":
    # If you want to run with sample data for testing:
    # kaggle_training_example()
    
    # If you want to run with your actual data:
    main()