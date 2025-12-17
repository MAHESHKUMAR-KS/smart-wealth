"""
AI Risk Persona Engine - Dynamic Investor Persona Classification

This module uses unsupervised machine learning (KMeans clustering) to classify
investors into behavioral personas based on their profile characteristics.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Dict
import warnings
warnings.filterwarnings('ignore')


class AIRiskPersonaEngine:
    """AI-powered investor persona classification engine"""
    
    def __init__(self):
        """Initialize the persona engine with mappings and prepare for clustering"""
        # Define persona mappings with explanations and allocation ranges
        self.persona_mappings = {
            0: {
                'name': 'Capital Preservation Planner',
                'explanation': 'Conservative investors focused on protecting their principal investment with minimal risk exposure.',
                'allocation_range': {'equity': (0, 20), 'debt': (60, 100)},
                'volatility_tolerance': 'Low'
            },
            1: {
                'name': 'Income-Focused Stabilizer',
                'explanation': 'Moderate-risk investors seeking regular income while maintaining stability in their investments.',
                'allocation_range': {'equity': (20, 40), 'debt': (40, 70)},
                'volatility_tolerance': 'Low-Medium'
            },
            2: {
                'name': 'Long-Term Growth Optimizer',
                'explanation': 'Balanced investors with a long-term perspective who accept moderate volatility for growth.',
                'allocation_range': {'equity': (40, 70), 'debt': (20, 50)},
                'volatility_tolerance': 'Medium'
            },
            3: {
                'name': 'Opportunistic Risk Taker',
                'explanation': 'Aggressive investors willing to accept high volatility for maximum growth potential.',
                'allocation_range': {'equity': (70, 100), 'debt': (0, 30)},
                'volatility_tolerance': 'High'
            }
        }
        
        # Categorical encodings
        self.risk_reaction_encoding = {
            'very_concerned': 0,
            'concerned': 1,
            'neutral': 2,
            'comfortable': 3,
            'very_comfortable': 4
        }
        
        self.goal_encoding = {
            'capital_preservation': 0,
            'income_generation': 1,
            'balanced_growth': 2,
            'aggressive_growth': 3
        }
        
        self.volatility_encoding = {
            'uncomfortable': 0,
            'slightly_uncomfortable': 1,
            'neutral': 2,
            'comfortable': 3,
            'very_comfortable': 4
        }
        
        # Initialize model components
        self.scaler = StandardScaler()
        self.model = None
        self.is_trained = False
        
        # Generate synthetic training data and train model
        self._generate_synthetic_data()
        self._train_model()
    
    def _generate_synthetic_data(self):
        """Generate synthetic investor profile data for training"""
        np.random.seed(42)  # For reproducible results
        
        # Generate realistic synthetic data
        n_samples = 40
        
        ages = np.random.randint(20, 70, n_samples)
        horizons = np.random.randint(1, 30, n_samples)
        risk_reactions = np.random.choice(list(self.risk_reaction_encoding.values()), n_samples)
        goals = np.random.choice(list(self.goal_encoding.values()), n_samples)
        volatilities = np.random.choice(list(self.volatility_encoding.values()), n_samples)
        
        # Create DataFrame
        self.training_data = pd.DataFrame({
            'age': ages,
            'horizon': horizons,
            'risk_reaction': risk_reactions,
            'goal': goals,
            'volatility': volatilities
        })
    
    def _train_model(self):
        """Train the KMeans clustering model"""
        # Prepare features for clustering
        features = self.training_data[['age', 'horizon', 'risk_reaction', 'goal', 'volatility']]
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        # Train KMeans model with 4 clusters
        self.model = KMeans(n_clusters=4, random_state=42, n_init=10)
        self.model.fit(scaled_features)
        
        self.is_trained = True
    
    def _encode_inputs(self, age: int, horizon: int, risk_reaction: str, 
                      goal: str, volatility: str) -> np.array:
        """Encode categorical inputs to numerical values"""
        # Encode categorical variables
        encoded_risk = self.risk_reaction_encoding.get(risk_reaction.lower(), 2)  # Default neutral
        encoded_goal = self.goal_encoding.get(goal.lower(), 2)  # Default balanced
        encoded_volatility = self.volatility_encoding.get(volatility.lower(), 2)  # Default neutral
        
        return np.array([[age, horizon, encoded_risk, encoded_goal, encoded_volatility]])
    
    def predict_persona(self, age: int, horizon: int, risk_reaction: str, 
                       goal: str, volatility: str) -> int:
        """Predict the investor persona cluster"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # Encode inputs
        encoded_input = self._encode_inputs(age, horizon, risk_reaction, goal, volatility)
        
        # Scale input
        scaled_input = self.scaler.transform(encoded_input)
        
        # Predict cluster
        cluster = self.model.predict(scaled_input)[0]
        
        return cluster


# Global instance for easy access
_persona_engine = AIRiskPersonaEngine()


def get_investor_persona(age: int, horizon: int, risk_reaction: str, 
                        goal: str, volatility: str) -> Dict:
    """
    Get investor persona based on profile characteristics using AI clustering.
    
    Args:
        age (int): Investor age
        horizon (int): Investment horizon in years
        risk_reaction (str): How investor reacts to market downturns
            Options: 'very_concerned', 'concerned', 'neutral', 'comfortable', 'very_comfortable'
        goal (str): Primary financial goal
            Options: 'capital_preservation', 'income_generation', 'balanced_growth', 'aggressive_growth'
        volatility (str): Comfort level with portfolio fluctuations
            Options: 'uncomfortable', 'slightly_uncomfortable', 'neutral', 'comfortable', 'very_comfortable'
        
    Returns:
        dict: Persona information including:
            - name: Persona name
            - explanation: Description of the persona
            - allocation_range: Suggested equity-debt allocation range
            - volatility_tolerance: Tolerance level for portfolio fluctuations
            - cluster_id: Internal cluster identifier
            
    Example:
        >>> persona = get_investor_persona(
        ...     age=30,
        ...     horizon=25,
        ...     risk_reaction="very_comfortable",
        ...     goal="aggressive_growth",
        ...     volatility="very_comfortable"
        ... )
        >>> print(persona['name'])
        Opportunistic Risk Taker
    """
    # Validate inputs
    if not isinstance(age, int) or age < 18 or age > 100:
        raise ValueError("Age must be an integer between 18 and 100")
    
    if not isinstance(horizon, int) or horizon < 1 or horizon > 50:
        raise ValueError("Horizon must be an integer between 1 and 50 years")
    
    # Validate categorical inputs
    if risk_reaction.lower() not in _persona_engine.risk_reaction_encoding:
        raise ValueError(f"Invalid risk_reaction. Must be one of {list(_persona_engine.risk_reaction_encoding.keys())}")
    
    if goal.lower() not in _persona_engine.goal_encoding:
        raise ValueError(f"Invalid goal. Must be one of {list(_persona_engine.goal_encoding.keys())}")
    
    if volatility.lower() not in _persona_engine.volatility_encoding:
        raise ValueError(f"Invalid volatility. Must be one of {list(_persona_engine.volatility_encoding.keys())}")
    
    # Get persona cluster prediction
    cluster = _persona_engine.predict_persona(age, horizon, risk_reaction, goal, volatility)
    
    # Return persona information
    persona_info = _persona_engine.persona_mappings[cluster].copy()
    persona_info['cluster_id'] = cluster
    
    return persona_info


# Integration example - how to use in fund_analyzer.py or portfolio_tools.py
"""
Example Integration:

In fund_analyzer.py or portfolio_tools.py, you can import and use the function like this:

from ai_risk_persona import get_investor_persona

# Get investor persona
persona = get_investor_persona(
    age=user_age,
    horizon=investment_horizon,
    risk_reaction=user_risk_reaction,
    goal=financial_goal,
    volatility=volatility_tolerance
)

# Access persona information
print(f"Investor Persona: {persona['name']}")
print(f"Explanation: {persona['explanation']}")
print(f"Suggested Equity Allocation: {persona['allocation_range']['equity'][0]}-{persona['allocation_range']['equity'][1]}%")
print(f"Volatility Tolerance: {persona['volatility_tolerance']}")

# Use persona information for fund recommendations
risk_profile_mapping = {
    'Capital Preservation Planner': 'conservative',
    'Income-Focused Stabilizer': 'moderate', 
    'Long-Term Growth Optimizer': 'moderate',
    'Opportunistic Risk Taker': 'aggressive'
}

mapped_risk_profile = risk_profile_mapping[persona['name']]
recommended_funds = fund_analyzer.recommend_funds(
    investment_amount=investment_amount,
    risk_profile=mapped_risk_profile,
    age=user_age,
    horizon=investment_horizon
)
"""