"""
AI-Powered Fund Performance Predictor
Uses machine learning to forecast mutual fund performance
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from typing import Dict, List, Tuple
from datetime import datetime, timedelta


class FundPerformancePredictor:
    """ML-based fund performance prediction engine"""
    
    def __init__(self):
        """Initialize the predictor with a Random Forest model"""
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.is_trained = False
        self.feature_columns = [
            'returns_1yr', 'returns_3yr', 'returns_5yr',
            'sharpe', 'sortino', 'alpha', 'beta', 'sd',
            'expense_ratio', 'fund_age_yr', 'rating'
        ]
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for training/prediction
        """
        # Select numerical features and handle missing values
        features = df[self.feature_columns].copy()
        
        # Fill missing values
        for col in features.columns:
            if col != 'rating':
                features[col] = features[col].fillna(features[col].median())
            else:
                features[col] = features[col].fillna(3)  # Default rating
        
        return features
    
    def train(self, df: pd.DataFrame, target_column: str = 'returns_3yr') -> Dict:
        """
        Train the model on historical data
        """
        # Prepare features
        X = self.prepare_features(df)
        y = df[target_column].fillna(df[target_column].median())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Feature importance
        feature_importance = dict(zip(X.columns, self.model.feature_importances_))
        
        return {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'r2_score': r2,
            'feature_importance': feature_importance
        }
    
    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Predict future performance for funds
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Prepare features
        X = self.prepare_features(df)
        
        # Make predictions
        predictions = self.model.predict(X)
        
        return pd.Series(predictions, index=df.index, name='predicted_returns')
    
    def save_model(self, filepath: str):
        """Save trained model to disk"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        joblib.dump(self.model, filepath)
    
    def load_model(self, filepath: str):
        """Load trained model from disk"""
        self.model = joblib.load(filepath)
        self.is_trained = True


class SentimentAnalyzer:
    """Simple sentiment analysis for financial news"""
    
    def __init__(self):
        # Simple keyword-based sentiment analysis
        self.positive_keywords = {
            'surge', 'rise', 'increase', 'boost', 'gain', 'profit', 'up', 
            'bullish', 'positive', 'growth', 'exceed', 'beat', 'outperform'
        }
        self.negative_keywords = {
            'fall', 'drop', 'decline', 'loss', 'crash', 'plunge', 'down',
            'bearish', 'negative', 'slump', 'miss', 'underperform', 'volatile'
        }
    
    def analyze_sentiment(self, text: str) -> float:
        """
        Analyze sentiment of text (returns score between -1 and 1)
        """
        text_lower = text.lower()
        positive_count = sum(1 for word in self.positive_keywords if word in text_lower)
        negative_count = sum(1 for word in self.negative_keywords if word in text_lower)
        
        # Simple scoring
        total = positive_count + negative_count
        if total == 0:
            return 0.0
        
        return (positive_count - negative_count) / total


class PortfolioOptimizer:
    """AI-powered portfolio optimization using reinforcement learning concepts"""
    
    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer()
    
    def optimize_allocation(self, 
                          current_portfolio: pd.DataFrame, 
                          market_sentiment: float = 0.0,
                          risk_appetite: str = 'moderate') -> Dict:
        """
        Optimize portfolio allocation based on predictions and market sentiment
        """
        # Adjust allocations based on market sentiment
        current_allocation = current_portfolio.groupby('category')['allocation_percentage'].sum().to_dict()
        
        # Base adjustments
        adjustments = {
            'Equity': 0,
            'Debt': 0,
            'Hybrid': 0
        }
        
        # Modify based on sentiment
        if market_sentiment > 0.3:  # Positive sentiment
            adjustments['Equity'] = 5
            adjustments['Debt'] = -3
            adjustments['Hybrid'] = -2
        elif market_sentiment < -0.3:  # Negative sentiment
            adjustments['Equity'] = -5
            adjustments['Debt'] = 3
            adjustments['Hybrid'] = 2
        
        # Risk appetite adjustments
        if risk_appetite == 'conservative':
            adjustments['Equity'] -= 3
            adjustments['Debt'] += 3
        elif risk_appetite == 'aggressive':
            adjustments['Equity'] += 3
            adjustments['Debt'] -= 3
        
        # Apply adjustments
        new_allocation = {}
        total_adjustment = sum(adjustments.values())
        
        for category, current_pct in current_allocation.items():
            adjustment = adjustments.get(category, 0)
            # Ensure percentages stay within reasonable bounds
            new_pct = max(0, min(100, current_pct + adjustment))
            new_allocation[category] = round(new_pct, 2)
        
        # Normalize to 100%
        total = sum(new_allocation.values())
        if total > 0:
            for category in new_allocation:
                new_allocation[category] = round(new_allocation[category] * 100 / total, 2)
        
        return {
            'current_allocation': current_allocation,
            'suggested_allocation': new_allocation,
            'market_sentiment': market_sentiment,
            'adjustments_made': adjustments
        }


# Example usage function
def integrate_ai_features(analyzer, df):
    """
    Example of how to integrate AI features into the existing system
    """
    # Initialize components
    predictor = FundPerformancePredictor()
    optimizer = PortfolioOptimizer()
    
    # Train predictor if not already trained
    if not predictor.is_trained:
        try:
            metrics = predictor.train(df)
            print(f"Model trained - R² Score: {metrics['r2_score']:.3f}")
        except Exception as e:
            print(f"Could not train model: {e}")
            return None, None
    
    # Make predictions
    try:
        predictions = predictor.predict(df)
        df['predicted_returns'] = predictions
        
        # Add to the main dataframe for use in recommendations
        analyzer.df['predicted_returns'] = predictions
        
        return predictor, optimizer
    except Exception as e:
        print(f"Could not make predictions: {e}")
        return None, None


if __name__ == "__main__":
    # Example usage
    print("AI Predictor module ready for integration")