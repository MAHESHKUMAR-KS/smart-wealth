"""
Kaggle Notebook for Training AI Risk Assessment Models
This script can be used in a Kaggle notebook environment
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import warnings
warnings.filterwarnings('ignore')

# For Kaggle, we'll create a synthetic dataset since we don't have access to real user data
def create_synthetic_dataset(n_samples=1000):
    """
    Create a synthetic dataset for training the models
    """
    np.random.seed(42)
    
    # Generate synthetic user data
    ages = np.random.randint(18, 70, n_samples)
    investment_amounts = np.random.lognormal(10, 1, n_samples)
    horizons = np.random.randint(1, 30, n_samples)
    
    # Risk reactions based on age (older people tend to be more conservative)
    risk_reaction_scores = []
    for age in ages:
        if age < 30:
            # Younger people more likely to buy more during drops
            risk_reaction_scores.append(np.random.choice([1, 2, 3, 4], p=[0.1, 0.1, 0.3, 0.5]))
        elif age < 50:
            # Middle-aged more balanced
            risk_reaction_scores.append(np.random.choice([1, 2, 3, 4], p=[0.2, 0.3, 0.3, 0.2]))
        else:
            # Older people more conservative
            risk_reaction_scores.append(np.random.choice([1, 2, 3, 4], p=[0.5, 0.3, 0.2, 0.0]))
    
    # Return preferences based on horizon (longer horizons prefer higher returns)
    return_preferences = []
    for horizon in horizons:
        if horizon < 5:
            return_preferences.append(np.random.choice([1, 1.5, 2], p=[0.6, 0.3, 0.1]))
        elif horizon < 10:
            return_preferences.append(np.random.choice([1.5, 2, 2.5], p=[0.3, 0.5, 0.2]))
        else:
            return_preferences.append(np.random.choice([2, 2.5, 3, 3.5], p=[0.1, 0.3, 0.4, 0.2]))
    
    # Market knowledge based on investment amount
    market_knowledge = []
    for amount in investment_amounts:
        if amount < 50000:
            market_knowledge.append(np.random.uniform(1, 2))
        elif amount < 100000:
            market_knowledge.append(np.random.uniform(2, 3))
        else:
            market_knowledge.append(np.random.uniform(3, 5))
    
    # Investment frequency based on horizon
    investment_frequency = horizons / np.random.uniform(1, 3, n_samples)
    investment_frequency = np.clip(investment_frequency, 1, 5)
    
    # Generate risk profiles based on features
    risk_profiles = []
    for i in range(n_samples):
        score = (risk_reaction_scores[i] + 
                return_preferences[i] + 
                market_knowledge[i]/2 +
                (70 - ages[i])/10) / 4
        
        if score < 2:
            risk_profiles.append('conservative')
        elif score < 3:
            risk_profiles.append('moderate')
        else:
            risk_profiles.append('aggressive')
    
    # Create DataFrame
    data = pd.DataFrame({
        'age': ages,
        'investment_amount': investment_amounts,
        'horizon': horizons,
        'risk_reaction_score': risk_reaction_scores,
        'return_preference_score': return_preferences,
        'market_knowledge_score': market_knowledge,
        'investment_frequency': investment_frequency,
        'risk_profile': risk_profiles
    })
    
    return data

# 1. User Behavior Analyzer (Random Forest Classifier)
class KaggleUserBehaviorAnalyzer:
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
        """Train the behavior analysis model"""
        # Prepare features and target
        X = self.prepare_features(df)
        y = df['risk_profile']
        
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
        
        # Feature importance
        feature_names = ['age', 'investment_amount', 'horizon', 
                        'risk_reaction_score', 'return_preference_score',
                        'market_knowledge_score', 'investment_frequency']
        feature_importance = dict(zip(feature_names, self.model.feature_importances_))
        
        return {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'feature_importance': feature_importance
        }
    
    def predict(self, user_data):
        """Predict risk profile for a user"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        # Prepare features
        features_df = pd.DataFrame([user_data])
        features_scaled = self.scaler.transform(features_df)
        
        # Predict
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        confidence = max(probabilities)
        
        return prediction, confidence
    
    def save_model(self, filename):
        """Save trained model"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler
            }, f)

# 2. User Cluster Analyzer (K-Means Clustering)
class KaggleUserClusterAnalyzer:
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=n_clusters, random_state=42)
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def prepare_features(self, df):
        """Prepare features for clustering"""
        # Numerical risk score
        risk_scores = {'conservative': 1, 'moderate': 2, 'aggressive': 3}
        df['risk_score'] = df['risk_profile'].map(risk_scores)
        
        # Return expectation
        df['return_expectation'] = df['risk_score'] * 3 + (df['horizon'] * 0.5)
        df['return_expectation'] = np.clip(df['return_expectation'], 5, 20)
        
        features = df[['age', 'investment_amount', 'horizon', 
                      'risk_score', 'return_expectation']]
        return features
    
    def fit(self, df):
        """Fit clustering model"""
        # Prepare features
        X = self.prepare_features(df)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit model
        cluster_labels = self.model.fit_predict(X_scaled)
        self.is_fitted = True
        
        # Add cluster labels to data for analysis
        df['cluster'] = cluster_labels
        
        # Calculate cluster statistics
        cluster_stats = {}
        for i in range(self.n_clusters):
            cluster_data = df[df['cluster'] == i]
            cluster_stats[f'cluster_{i}'] = {
                'size': len(cluster_data),
                'avg_age': cluster_data['age'].mean(),
                'avg_investment': cluster_data['investment_amount'].mean(),
                'avg_horizon': cluster_data['horizon'].mean(),
                'avg_risk_score': cluster_data['risk_score'].mean(),
                'avg_return_expectation': cluster_data['return_expectation'].mean()
            }
        
        return {
            'cluster_stats': cluster_stats,
            'inertia': self.model.inertia_,
            'n_clusters': self.n_clusters
        }
    
    def predict_cluster(self, user_data):
        """Assign user to cluster"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        # Convert risk profile to score
        risk_scores = {'conservative': 1, 'moderate': 2, 'aggressive': 3}
        user_data['risk_score'] = risk_scores[user_data['risk_profile']]
        user_data['return_expectation'] = user_data['risk_score'] * 3 + (user_data['horizon'] * 0.5)
        user_data['return_expectation'] = max(5, min(20, user_data['return_expectation']))
        
        # Prepare features
        features_df = pd.DataFrame([user_data])
        features = features_df[['age', 'investment_amount', 'horizon', 
                               'risk_score', 'return_expectation']]
        features_scaled = self.scaler.transform(features)
        
        # Predict cluster
        cluster = self.model.predict(features_scaled)[0]
        return cluster
    
    def save_model(self, filename):
        """Save fitted model"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'n_clusters': self.n_clusters
            }, f)

# Main execution for Kaggle
def main():
    """
    Main function for Kaggle notebook
    """
    print("AI Risk Assessment Model Training on Kaggle")
    print("=" * 50)
    
    # Step 1: Create synthetic dataset
    print("\n1. Creating synthetic dataset...")
    df = create_synthetic_dataset(1000)
    print(f"Dataset created with {len(df)} samples")
    print("\nDataset Info:")
    print(df.head())
    print(f"\nRisk Profile Distribution:")
    print(df['risk_profile'].value_counts())
    
    # Step 2: Visualize data
    print("\n2. Visualizing data...")
    
    # Plot risk profile distribution
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    df['risk_profile'].value_counts().plot(kind='bar')
    plt.title('Risk Profile Distribution')
    plt.xlabel('Risk Profile')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    # Plot age vs investment amount by risk profile
    plt.subplot(1, 2, 2)
    for profile in df['risk_profile'].unique():
        subset = df[df['risk_profile'] == profile]
        plt.scatter(subset['age'], subset['investment_amount'], 
                   label=profile, alpha=0.6)
    plt.xlabel('Age')
    plt.ylabel('Investment Amount')
    plt.title('Age vs Investment Amount by Risk Profile')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Step 3: Train User Behavior Analyzer
    print("\n3. Training User Behavior Analyzer (Random Forest)...")
    behavior_analyzer = KaggleUserBehaviorAnalyzer()
    behavior_results = behavior_analyzer.train(df)
    
    print(f"Behavior Model Accuracy: {behavior_results['accuracy']:.3f}")
    print("\nFeature Importance:")
    for feature, importance in behavior_results['feature_importance'].items():
        print(f"  {feature}: {importance:.3f}")
    
    # Step 4: Train User Cluster Analyzer
    print("\n4. Training User Cluster Analyzer (K-Means)...")
    cluster_analyzer = KaggleUserClusterAnalyzer(n_clusters=5)
    cluster_results = cluster_analyzer.fit(df)
    
    print(f"Clustering Inertia: {cluster_results['inertia']:.2f}")
    print(f"Number of clusters: {cluster_results['n_clusters']}")
    
    print("\nCluster Statistics:")
    for cluster_name, stats in cluster_results['cluster_stats'].items():
        print(f"  {cluster_name}:")
        print(f"    Size: {stats['size']}")
        print(f"    Avg Age: {stats['avg_age']:.1f}")
        print(f"    Avg Investment: ${stats['avg_investment']:,.0f}")
        print(f"    Avg Horizon: {stats['avg_horizon']:.1f} years")
    
    # Step 5: Test with sample users
    print("\n5. Testing with sample users...")
    
    # Sample users
    sample_users = [
        {
            'age': 25, 'investment_amount': 30000, 'horizon': 15,
            'risk_reaction_score': 4, 'return_preference_score': 3.5,
            'market_knowledge_score': 2.0, 'investment_frequency': 4.0,
            'risk_profile': 'aggressive'
        },
        {
            'age': 45, 'investment_amount': 150000, 'horizon': 8,
            'risk_reaction_score': 2, 'return_preference_score': 2.5,
            'market_knowledge_score': 4.0, 'investment_frequency': 2.0,
            'risk_profile': 'moderate'
        },
        {
            'age': 55, 'investment_amount': 200000, 'horizon': 5,
            'risk_reaction_score': 1, 'return_preference_score': 1.5,
            'market_knowledge_score': 4.5, 'investment_frequency': 1.5,
            'risk_profile': 'conservative'
        }
    ]
    
    for i, user in enumerate(sample_users):
        print(f"\nSample User {i+1}:")
        print(f"  Age: {user['age']}, Investment: ${user['investment_amount']:,.0f}")
        print(f"  Horizon: {user['horizon']} years")
        
        # Predict risk profile
        predicted_profile, confidence = behavior_analyzer.predict(user)
        print(f"  Predicted Risk Profile: {predicted_profile} (Confidence: {confidence:.2f})")
        
        # Assign to cluster
        cluster = cluster_analyzer.predict_cluster(user)
        print(f"  Assigned Cluster: {cluster}")
    
    # Step 6: Save models
    print("\n6. Saving trained models...")
    behavior_analyzer.save_model('behavior_model.pkl')
    cluster_analyzer.save_model('cluster_model.pkl')
    print("Models saved successfully!")
    
    print("\n" + "=" * 50)
    print("Training completed successfully!")
    print("Download the saved models for use in your application.")

# For direct execution
if __name__ == "__main__":
    main()