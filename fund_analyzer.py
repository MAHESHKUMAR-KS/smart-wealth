"""
Intelligent Fund Analysis and Recommendation Engine
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from config import *
from ai_predictor import FundPerformancePredictor
from ai_risk_assessment import EnhancedRiskProfiler

# Optional import of AI persona engine for enhanced functionality
try:
    from ai_risk_persona import get_investor_persona
    AI_PERSONA_AVAILABLE = True
except ImportError:
    AI_PERSONA_AVAILABLE = False


class FundAnalyzer:
    """Advanced fund analysis with multi-factor scoring"""
    
    def __init__(self, data_path: str = DATA_PATH):
        """Initialize with fund data"""
        self.df = self.load_and_clean_data(data_path)
        
    def load_and_clean_data(self, path: str) -> pd.DataFrame:
        """Load and clean fund data with proper error handling"""
        df = pd.read_csv(path)
        
        # Replace '-' with NaN and convert to numeric
        numeric_cols = ['sharpe', 'beta', 'alpha', 'sortino', 'sd', 'expense_ratio',
                       'returns_1yr', 'returns_3yr', 'returns_5yr', 'rating', 'fund_age_yr']
        
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove funds with too many missing critical metrics
        df = df.dropna(subset=['sharpe', 'alpha', 'sortino', 'returns_1yr', 'category'])
        
        return df
    
    def calculate_quality_score(self, row: pd.Series) -> float:
        """
        Calculate comprehensive quality score (0-100)
        Factors: Sharpe, Sortino, Alpha, Returns, Rating, Expense Ratio
        """
        score = 0
        
        # Sharpe Ratio (0-25 points)
        if pd.notna(row['sharpe']):
            score += min(row['sharpe'] / 2 * 25, 25)
        
        # Sortino Ratio (0-25 points)
        if pd.notna(row['sortino']):
            score += min(row['sortino'] / 2 * 25, 25)
        
        # Alpha (0-20 points)
        if pd.notna(row['alpha']):
            score += min(max(row['alpha'], 0) / 5 * 20, 20)
        
        # Rating (0-15 points)
        if pd.notna(row['rating']):
            score += row['rating'] / 5 * 15
        
        # Consistency - 3yr returns (0-10 points)
        if pd.notna(row['returns_3yr']):
            score += min(max(row['returns_3yr'], 0) / 30 * 10, 10)
        
        # Low expense ratio bonus (0-5 points)
        if pd.notna(row['expense_ratio']):
            score += max(5 - row['expense_ratio'], 0)
        
        return round(score, 2)
    
    def calculate_risk_score(self, row: pd.Series) -> float:
        """
        Calculate risk score (0-100, higher = riskier)
        Factors: Beta, Standard Deviation, Risk Level
        """
        risk = 0
        
        # Beta (0-40 points)
        if pd.notna(row['beta']) and row['beta'] != 0:
            risk += min(abs(row['beta']) * 40, 40)
        else:
            risk += 20  # Default medium risk if beta missing
        
        # Standard Deviation (0-40 points)
        if pd.notna(row['sd']):
            risk += min(row['sd'] / 25 * 40, 40)
        else:
            risk += 20  # Default medium risk if SD missing
        
        # Category risk level (0-20 points)
        if pd.notna(row['risk_level']):
            risk += row['risk_level'] / 6 * 20
        
        return round(risk, 2)
    
    def get_category_allocation(self, risk_profile: str, age: int, horizon: int) -> Dict[str, float]:
        """
        Calculate optimal category allocation based on multiple factors
        """
        profile = RISK_PROFILES[risk_profile]
        
        # Base allocation from risk profile
        equity_min, equity_max = profile['equity_range']
        debt_min, debt_max = profile['debt_range']
        hybrid_min, hybrid_max = profile['hybrid_range']
        
        # Age-based adjustment (Rule: 100 - age for equity)
        age_equity = max(20, min(100 - age, 80))
        
        # Horizon-based adjustment (longer horizon = more equity)
        horizon_factor = min(horizon / 20, 1.0)
        
        # Weighted calculation
        equity = (equity_min + equity_max) / 2 * 0.5 + age_equity * 0.3 + (horizon_factor * 60) * 0.2
        equity = max(equity_min, min(equity_max, equity))
        
        remaining = 100 - equity
        
        # Split remaining between debt and hybrid based on risk profile
        if risk_profile == 'conservative':
            debt = remaining * 0.7
            hybrid = remaining * 0.3
        elif risk_profile == 'moderate':
            debt = remaining * 0.5
            hybrid = remaining * 0.5
        else:  # aggressive
            debt = remaining * 0.4
            hybrid = remaining * 0.6
        
        return {
            'Equity': round(equity, 2),
            'Debt': round(debt, 2),
            'Hybrid': round(hybrid, 2)
        }
    
    def recommend_funds(self, 
                       investment_amount: float,
                       risk_profile: str,
                       age: int,
                       horizon: int,
                       goal: str = None,
                       exclude_amcs: List[str] = None) -> pd.DataFrame:
        """
        Intelligent fund recommendation with diversification and AI predictions
        """
        # Calculate allocation
        allocation = self.get_category_allocation(risk_profile, age, horizon)
        
        # Calculate quality and risk scores
        self.df['quality_score'] = self.df.apply(self.calculate_quality_score, axis=1)
        self.df['risk_score'] = self.df.apply(self.calculate_risk_score, axis=1)
        
        # Use AI predictions if available
        if 'predicted_returns' in self.df.columns:
            # Incorporate predicted returns into quality score
            pred_weight = 0.2  # 20% weight to predictions
            max_pred = self.df['predicted_returns'].max()
            if max_pred > 0:
                pred_score = (self.df['predicted_returns'] / max_pred) * 100
                self.df['quality_score'] = (
                    self.df['quality_score'] * (1 - pred_weight) + 
                    pred_score * pred_weight
                )
        
        recommended_funds = []
        
        for category, percentage in allocation.items():
            if percentage < 5:  # Skip if allocation too small
                continue
                
            amount = investment_amount * (percentage / 100)
            
            # Filter funds by category
            category_funds = self.df[self.df['category'] == category].copy()
            
            # Exclude specific AMCs if requested
            if exclude_amcs:
                category_funds = category_funds[~category_funds['amc_name'].isin(exclude_amcs)]
            
            # Apply quality filters based on risk profile
            quality_threshold = QUALITY_THRESHOLDS['good']
            category_funds = category_funds[
                (category_funds['sharpe'] >= quality_threshold['sharpe']) &
                (category_funds['sortino'] >= quality_threshold['sortino']) &
                (category_funds['alpha'] >= quality_threshold['alpha'])
            ]
            
            # Sort by quality score
            category_funds = category_funds.sort_values('quality_score', ascending=False)
            
            # Diversification: Select funds from different AMCs
            selected = self._diversify_selection(category_funds, min(3, len(category_funds)))
            
            for _, fund in selected.iterrows():
                recommended_funds.append({
                    'scheme_name': fund['scheme_name'],
                    'category': fund['category'],
                    'amc_name': fund['amc_name'],
                    'allocation_percentage': percentage / len(selected),
                    'investment_amount': amount / len(selected),
                    'quality_score': fund['quality_score'],
                    'risk_score': fund['risk_score'],
                    'sharpe': fund['sharpe'],
                    'sortino': fund['sortino'],
                    'alpha': fund['alpha'],
                    'beta': fund['beta'],
                    'sd': fund['sd'],
                    'returns_1yr': fund['returns_1yr'],
                    'returns_3yr': fund['returns_3yr'],
                    'returns_5yr': fund['returns_5yr'],
                    'expense_ratio': fund['expense_ratio'],
                    'rating': fund['rating'],
                    'fund_manager': fund['fund_manager']
                })
        
        result_df = pd.DataFrame(recommended_funds)
        
        # Ensure total allocation is 100%
        if len(result_df) > 0:
            total = result_df['allocation_percentage'].sum()
            result_df['allocation_percentage'] = result_df['allocation_percentage'] / total * 100
        
        return result_df
    
    def recommend_funds_with_ai_persona(self,
                                       investment_amount: float,
                                       age: int,
                                       horizon: int,
                                       risk_reaction: str,
                                       goal: str,
                                       volatility: str,
                                       exclude_amcs: List[str] = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Enhanced fund recommendation using AI investor persona engine
        
        Args:
            investment_amount (float): Total investment amount
            age (int): Investor age
            horizon (int): Investment horizon in years
            risk_reaction (str): How investor reacts to market downturns
            goal (str): Primary financial goal
            volatility (str): Comfort level with portfolio fluctuations
            exclude_amcs (List[str]): AMCs to exclude from recommendations
            
        Returns:
            Tuple[pd.DataFrame, Dict]: Recommended funds and persona information
        """
        if not AI_PERSONA_AVAILABLE:
            raise ImportError("AI Risk Persona Engine not available")
        
        # Get AI investor persona
        persona = get_investor_persona(age, horizon, risk_reaction, goal, volatility)
        
        # Map persona to traditional risk profile
        persona_risk_mapping = {
            'Capital Preservation Planner': 'conservative',
            'Income-Focused Stabilizer': 'moderate', 
            'Long-Term Growth Optimizer': 'moderate',
            'Opportunistic Risk Taker': 'aggressive'
        }
        
        mapped_risk_profile = persona_risk_mapping[persona['name']]
        
        # Get recommendations using traditional method with AI-enhanced risk profile
        recommendations = self.recommend_funds(
            investment_amount=investment_amount,
            risk_profile=mapped_risk_profile,
            age=age,
            horizon=horizon,
            goal=goal,
            exclude_amcs=exclude_amcs
        )
        
        return recommendations, persona
    
    def _diversify_selection(self, funds: pd.DataFrame, n: int) -> pd.DataFrame:
        """Select funds ensuring AMC diversification"""
        selected = []
        used_amcs = set()
        
        for _, fund in funds.iterrows():
            if len(selected) >= n:
                break
            
            # Prefer different AMCs
            if fund['amc_name'] not in used_amcs or len(funds[~funds['amc_name'].isin(used_amcs)]) == 0:
                selected.append(fund)
                used_amcs.add(fund['amc_name'])
        
        return pd.DataFrame(selected)
    
    def compare_funds(self, fund_names: List[str]) -> pd.DataFrame:
        """Detailed comparison of specific funds"""
        funds = self.df[self.df['scheme_name'].isin(fund_names)].copy()
        
        if len(funds) == 0:
            return pd.DataFrame()
        
        funds['quality_score'] = funds.apply(self.calculate_quality_score, axis=1)
        funds['risk_score'] = funds.apply(self.calculate_risk_score, axis=1)
        
        comparison_cols = ['scheme_name', 'category', 'amc_name', 'quality_score', 
                          'risk_score', 'sharpe', 'sortino', 'alpha', 'beta', 
                          'returns_1yr', 'returns_3yr', 'returns_5yr', 
                          'expense_ratio', 'rating', 'fund_manager']
        
        return funds[comparison_cols]
    
    def get_top_performers(self, category: str = None, n: int = 10) -> pd.DataFrame:
        """Get top performing funds by category"""
        funds = self.df.copy()
        
        if category:
            funds = funds[funds['category'] == category]
        
        funds['quality_score'] = funds.apply(self.calculate_quality_score, axis=1)
        
        return funds.nlargest(n, 'quality_score')[
            ['scheme_name', 'category', 'amc_name', 'quality_score', 
             'sharpe', 'alpha', 'returns_1yr', 'returns_3yr', 'rating']
        ]
    
    def get_fund_statistics(self) -> Dict:
        """Get overall fund database statistics"""
        return {
            'total_funds': len(self.df),
            'categories': self.df['category'].value_counts().to_dict(),
            'amcs': self.df['amc_name'].nunique(),
            'avg_returns_1yr': self.df['returns_1yr'].mean(),
            'avg_returns_3yr': self.df['returns_3yr'].mean(),
            'avg_expense_ratio': self.df['expense_ratio'].mean(),
            'top_rated_count': len(self.df[self.df['rating'] >= 4])
        }
