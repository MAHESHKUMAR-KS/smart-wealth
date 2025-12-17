"""
AI-Powered Risk Assessment System
Implements machine learning models to analyze user behavior patterns and refine risk profiling
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from typing import Dict, List, Tuple, Any
import json
import os


class UserBehaviorAnalyzer:
    """Analyzes user behavior patterns for risk profiling"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.behavior_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False
        self.behavior_features = [
            'age', 'investment_amount', 'horizon', 
            'risk_reaction_score', 'return_preference_score',
            'market_knowledge_score', 'investment_frequency'
        ]
    
    def extract_behavior_features(self, user_profile: Dict) -> Dict:
        """
        Extract behavioral features from user profile
        """
        features = {
            'age': user_profile.get('age', 30),
            'investment_amount': user_profile.get('investment_amount', 50000),
            'horizon': user_profile.get('horizon', 10),
            'risk_reaction_score': self._calculate_risk_reaction_score(user_profile),
            'return_preference_score': self._calculate_return_preference_score(user_profile),
            'market_knowledge_score': self._calculate_market_knowledge_score(user_profile),
            'investment_frequency': self._estimate_investment_frequency(user_profile)
        }
        
        return features
    
    def _calculate_risk_reaction_score(self, user_profile: Dict) -> float:
        """
        Calculate risk reaction score based on user responses
        """
        q1_response = user_profile.get('risk_reaction', 'Hold steady')
        risk_scores = {
            "Panic and sell": 1, 
            "Feel concerned": 2, 
            "Hold steady": 3, 
            "Buy more": 4
        }
        return risk_scores.get(q1_response, 3)
    
    def _calculate_return_preference_score(self, user_profile: Dict) -> float:
        """
        Calculate return preference score based on user goals
        """
        goal = user_profile.get('goal', 'wealth_creation')
        goal_scores = {
            'emergency_fund': 1,
            'short_term_savings': 1.5,
            'home_purchase': 2,
            'child_education': 2.5,
            'retirement': 3,
            'wealth_creation': 3.5
        }
        return goal_scores.get(goal, 3)
    
    def _calculate_market_knowledge_score(self, user_profile: Dict) -> float:
        """
        Estimate market knowledge based on investment experience
        """
        # This would typically come from additional questions
        # For now, we'll estimate based on age and investment amount
        age = user_profile.get('age', 30)
        investment_amount = user_profile.get('investment_amount', 50000)
        
        # Heuristic: older users and those with higher investments likely have more experience
        knowledge_score = min(5, max(1, (age / 10) + (investment_amount / 100000)))
        return knowledge_score
    
    def _estimate_investment_frequency(self, user_profile: Dict) -> float:
        """
        Estimate how frequently user invests based on profile
        """
        # This would come from actual user behavior tracking
        # For now, we'll use horizon as a proxy
        horizon = user_profile.get('horizon', 10)
        # Longer horizons might indicate more experienced/frequent investors
        return min(5, max(1, horizon / 2))
    
    def prepare_training_data(self, user_profiles: List[Dict]) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data from user profiles
        """
        features_list = []
        labels = []
        
        for profile in user_profiles:
            features = self.extract_behavior_features(profile)
            features_list.append(features)
            
            # Use existing risk profile as label for training
            risk_label = profile.get('risk_profile', 'moderate')
            labels.append(risk_label)
        
        X = pd.DataFrame(features_list)
        y = pd.Series(labels)
        
        return X, y
    
    def train_behavior_model(self, user_profiles: List[Dict]) -> Dict:
        """
        Train the behavior analysis model
        """
        if len(user_profiles) < 10:
            raise ValueError("Need at least 10 user profiles for training")
        
        X, y = self.prepare_training_data(user_profiles)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        self.behavior_model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate
        y_pred = self.behavior_model.predict(X_test)
        
        # Feature importance
        feature_importance = dict(zip(self.behavior_features, 
                                    self.behavior_model.feature_importances_))
        
        return {
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'feature_importance': feature_importance,
            'accuracy': self.behavior_model.score(X_test, y_test)
        }
    
    def predict_risk_profile(self, user_profile: Dict) -> Tuple[str, float]:
        """
        Predict risk profile for a user
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        features = self.extract_behavior_features(user_profile)
        features_df = pd.DataFrame([features])
        features_scaled = self.scaler.transform(features_df)
        
        # Predict
        prediction = self.behavior_model.predict(features_scaled)[0]
        probabilities = self.behavior_model.predict_proba(features_scaled)[0]
        confidence = max(probabilities)
        
        return prediction, confidence
    
    def save_model(self, filepath: str):
        """Save trained model to disk"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'scaler': self.scaler,
            'model': self.behavior_model,
            'features': self.behavior_features
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str):
        """Load trained model from disk"""
        model_data = joblib.load(filepath)
        self.scaler = model_data['scaler']
        self.behavior_model = model_data['model']
        self.behavior_features = model_data['features']
        self.is_trained = True


class UserClusterAnalyzer:
    """Clusters users with similar profiles for better recommendations"""
    
    def __init__(self, n_clusters: int = 5):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.cluster_model = KMeans(n_clusters=n_clusters, random_state=42)
        self.is_fitted = False
        self.cluster_features = [
            'age', 'investment_amount', 'horizon', 
            'risk_score', 'return_expectation'
        ]
    
    def extract_cluster_features(self, user_profile: Dict) -> Dict:
        """
        Extract features for clustering
        """
        features = {
            'age': user_profile.get('age', 30),
            'investment_amount': user_profile.get('investment_amount', 50000),
            'horizon': user_profile.get('horizon', 10),
            'risk_score': self._calculate_risk_score(user_profile),
            'return_expectation': self._calculate_return_expectation(user_profile)
        }
        
        return features
    
    def _calculate_risk_score(self, user_profile: Dict) -> float:
        """
        Calculate numerical risk score
        """
        risk_profile = user_profile.get('risk_profile', 'moderate')
        risk_scores = {
            'conservative': 1,
            'moderate': 2,
            'aggressive': 3
        }
        return risk_scores.get(risk_profile, 2)
    
    def _calculate_return_expectation(self, user_profile: Dict) -> float:
        """
        Estimate return expectation based on risk profile and horizon
        """
        risk_score = self._calculate_risk_score(user_profile)
        horizon = user_profile.get('horizon', 10)
        
        # Higher risk and longer horizon typically mean higher return expectations
        return_expectation = risk_score * 3 + (horizon * 0.5)
        return min(20, max(5, return_expectation))  # Reasonable range: 5-20%
    
    def prepare_clustering_data(self, user_profiles: List[Dict]) -> pd.DataFrame:
        """
        Prepare data for clustering
        """
        features_list = []
        
        for profile in user_profiles:
            features = self.extract_cluster_features(profile)
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def fit_clusters(self, user_profiles: List[Dict]) -> Dict:
        """
        Fit clustering model to user profiles
        """
        if len(user_profiles) < self.n_clusters:
            raise ValueError(f"Need at least {self.n_clusters} user profiles for clustering")
        
        X = self.prepare_clustering_data(user_profiles)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit clustering model
        cluster_labels = self.cluster_model.fit_predict(X_scaled)
        self.is_fitted = True
        
        # Add cluster labels to original data for analysis
        X['cluster'] = cluster_labels
        
        # Calculate cluster statistics
        cluster_stats = {}
        for i in range(self.n_clusters):
            cluster_data = X[X['cluster'] == i]
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
            'inertia': self.cluster_model.inertia_,
            'n_clusters': self.n_clusters
        }
    
    def assign_user_to_cluster(self, user_profile: Dict) -> int:
        """
        Assign a user to the most similar cluster
        """
        if not self.is_fitted:
            raise ValueError("Clustering model must be fitted before assigning users")
        
        features = self.extract_cluster_features(user_profile)
        features_df = pd.DataFrame([features])
        features_scaled = self.scaler.transform(features_df)
        
        # Predict cluster
        cluster = self.cluster_model.predict(features_scaled)[0]
        return cluster
    
    def get_similar_users(self, user_profile: Dict, all_profiles: List[Dict], 
                         max_users: int = 5) -> List[Dict]:
        """
        Find users most similar to the given user based on cluster assignment
        """
        if not self.is_fitted:
            raise ValueError("Clustering model must be fitted before finding similar users")
        
        user_cluster = self.assign_user_to_cluster(user_profile)
        
        # Find users in the same cluster
        similar_users = []
        for profile in all_profiles:
            profile_cluster = self.assign_user_to_cluster(profile)
            if profile_cluster == user_cluster:
                similar_users.append(profile)
                if len(similar_users) >= max_users:
                    break
        
        return similar_users
    
    def save_model(self, filepath: str):
        """Save fitted clustering model to disk"""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        model_data = {
            'scaler': self.scaler,
            'model': self.cluster_model,
            'features': self.cluster_features,
            'n_clusters': self.n_clusters
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str):
        """Load fitted clustering model from disk"""
        model_data = joblib.load(filepath)
        self.scaler = model_data['scaler']
        self.cluster_model = model_data['model']
        self.cluster_features = model_data['features']
        self.n_clusters = model_data['n_clusters']
        self.is_fitted = True


class EnhancedRiskProfiler:
    """Enhanced risk profiler combining traditional and AI approaches"""
    
    def __init__(self):
        self.behavior_analyzer = UserBehaviorAnalyzer()
        self.cluster_analyzer = UserClusterAnalyzer()
        self.user_profiles_database = []  # In production, this would be a database
    
    def add_user_profile(self, user_profile: Dict):
        """
        Add user profile to database for training and clustering
        """
        self.user_profiles_database.append(user_profile)
    
    def train_models(self) -> Dict:
        """
        Train both behavior analysis and clustering models
        """
        if len(self.user_profiles_database) < 10:
            return {
                'status': 'insufficient_data',
                'message': 'Need at least 10 user profiles for training'
            }
        
        # Train behavior model
        behavior_results = self.behavior_analyzer.train_behavior_model(
            self.user_profiles_database
        )
        
        # Fit clustering model
        try:
            cluster_results = self.cluster_analyzer.fit_clusters(
                self.user_profiles_database
            )
        except ValueError as e:
            cluster_results = {
                'status': 'error',
                'message': str(e)
            }
        
        return {
            'behavior_analysis': behavior_results,
            'clustering': cluster_results
        }
    
    def enhanced_risk_profile(self, user_profile: Dict) -> Dict:
        """
        Generate enhanced risk profile using AI models
        """
        results = {
            'traditional_profile': user_profile.get('risk_profile', 'moderate')
        }
        
        # AI-based risk prediction
        try:
            if self.behavior_analyzer.is_trained:
                ai_prediction, confidence = self.behavior_analyzer.predict_risk_profile(
                    user_profile
                )
                results['ai_predicted_profile'] = ai_prediction
                results['prediction_confidence'] = confidence
            else:
                results['ai_predicted_profile'] = None
                results['prediction_confidence'] = 0.0
        except Exception as e:
            results['ai_predicted_profile'] = None
            results['prediction_confidence'] = 0.0
            results['prediction_error'] = str(e)
        
        # Cluster assignment
        try:
            if self.cluster_analyzer.is_fitted:
                cluster = self.cluster_analyzer.assign_user_to_cluster(user_profile)
                results['cluster_assignment'] = cluster
                
                # Find similar users
                similar_users = self.cluster_analyzer.get_similar_users(
                    user_profile, self.user_profiles_database, max_users=3
                )
                results['similar_users_count'] = len(similar_users)
            else:
                results['cluster_assignment'] = None
                results['similar_users_count'] = 0
        except Exception as e:
            results['cluster_assignment'] = None
            results['similar_users_count'] = 0
            results['clustering_error'] = str(e)
        
        # Blend traditional and AI profiles
        if results['ai_predicted_profile'] and results['prediction_confidence'] > 0.7:
            results['final_profile'] = results['ai_predicted_profile']
        else:
            results['final_profile'] = results['traditional_profile']
        
        return results
    
    def save_models(self, base_path: str = "."):
        """Save trained models to disk"""
        try:
            self.behavior_analyzer.save_model(
                os.path.join(base_path, "behavior_model.pkl")
            )
            self.cluster_analyzer.save_model(
                os.path.join(base_path, "cluster_model.pkl")
            )
            return True
        except Exception as e:
            print(f"Error saving models: {e}")
            return False
    
    def load_models(self, base_path: str = "."):
        """Load trained models from disk"""
        try:
            self.behavior_analyzer.load_model(
                os.path.join(base_path, "behavior_model.pkl")
            )
            self.cluster_analyzer.load_model(
                os.path.join(base_path, "cluster_model.pkl")
            )
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False


# Example usage and testing
def demo_ai_risk_assessment():
    """
    Demonstrate the AI risk assessment system
    """
    # Initialize the enhanced risk profiler
    profiler = EnhancedRiskProfiler()
    
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
        }
    ]
    
    # Add profiles to database
    for profile in sample_profiles:
        profiler.add_user_profile(profile)
    
    # Train models
    print("Training AI models...")
    training_results = profiler.train_models()
    print(f"Training completed: {training_results}")
    
    # Test with a new user
    new_user = {
        'age': 29, 'investment_amount': 45000, 'horizon': 13,
        'risk_reaction': 'Buy more', 'goal': 'wealth_creation',
        'risk_profile': 'aggressive'  # Traditional profile
    }
    
    print("\nAnalyzing new user...")
    enhanced_profile = profiler.enhanced_risk_profile(new_user)
    print(f"Enhanced risk profile: {enhanced_profile}")
    
    return profiler


if __name__ == "__main__":
    # Run demo
    profiler = demo_ai_risk_assessment()
    
    # Save models
    profiler.save_models()
    print("\nModels saved successfully!")