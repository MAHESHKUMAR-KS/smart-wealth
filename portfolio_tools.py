"""
Portfolio Optimization and SIP Calculation Tools
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from config import TAX_RATES

# Optional import of AI persona engine for enhanced functionality
try:
    from ai_risk_persona import get_investor_persona
    AI_PERSONA_AVAILABLE = True
except ImportError:
    AI_PERSONA_AVAILABLE = False


class SIPCalculator:
    """Systematic Investment Plan Calculator with advanced features"""
    
    @staticmethod
    def calculate_future_value(monthly_sip: float, 
                               annual_return: float, 
                               years: int,
                               step_up_percentage: float = 0) -> Dict:
        """
        Calculate SIP future value with step-up option
        """
        months = years * 12
        monthly_rate = annual_return / 12
        
        future_value = 0
        total_invested = 0
        yearly_breakdown = []
        
        current_sip = monthly_sip
        
        for year in range(1, years + 1):
            year_fv = 0
            year_invested = 0
            
            for month in range(12):
                # Compound previous value
                future_value = future_value * (1 + monthly_rate)
                # Add current month SIP
                future_value += current_sip
                total_invested += current_sip
                
                year_fv += current_sip
                year_invested += current_sip
            
            yearly_breakdown.append({
                'year': year,
                'monthly_sip': round(current_sip, 2),
                'invested_this_year': round(year_invested, 2),
                'total_invested': round(total_invested, 2),
                'portfolio_value': round(future_value, 2),
                'gains': round(future_value - total_invested, 2)
            })
            
            # Apply step-up for next year
            if step_up_percentage > 0:
                current_sip = current_sip * (1 + step_up_percentage / 100)
        
        return {
            'future_value': round(future_value, 2),
            'total_invested': round(total_invested, 2),
            'total_gains': round(future_value - total_invested, 2),
            'returns_percentage': round((future_value - total_invested) / total_invested * 100, 2),
            'yearly_breakdown': yearly_breakdown
        }
    
    @staticmethod
    def calculate_required_sip(target_amount: float,
                              annual_return: float,
                              years: int,
                              step_up_percentage: float = 0) -> float:
        """
        Calculate monthly SIP required to reach target amount
        """
        # Binary search for required SIP
        low, high = 100, target_amount
        tolerance = 100
        
        while low < high:
            mid = (low + high) / 2
            result = SIPCalculator.calculate_future_value(mid, annual_return, years, step_up_percentage)
            
            if abs(result['future_value'] - target_amount) < tolerance:
                return round(mid, 2)
            elif result['future_value'] < target_amount:
                low = mid + 1
            else:
                high = mid - 1
        
        return round(low, 2)
    
    @staticmethod
    def calculate_tax_impact(gains: float, 
                            investment_type: str,
                            holding_period_years: float) -> Dict:
        """
        Calculate tax on mutual fund gains
        """
        if investment_type.lower() == 'equity':
            if holding_period_years >= 1:
                # LTCG - 10% above 1 lakh
                tax_free_limit = 100000
                taxable_gains = max(0, gains - tax_free_limit)
                tax = taxable_gains * TAX_RATES['ltcg_equity']
                tax_type = 'LTCG (Long Term Capital Gains)'
            else:
                # STCG - 15%
                tax = gains * TAX_RATES['stcg_equity']
                tax_type = 'STCG (Short Term Capital Gains)'
        else:  # Debt
            if holding_period_years >= 3:
                # LTCG with indexation (simplified to 20%)
                tax = gains * TAX_RATES['ltcg_debt']
                tax_type = 'LTCG with Indexation'
            else:
                # STCG - as per slab (assuming 30%)
                tax = gains * TAX_RATES['stcg_debt']
                tax_type = 'STCG (As per Tax Slab)'
        
        return {
            'gross_gains': round(gains, 2),
            'tax_amount': round(tax, 2),
            'net_gains': round(gains - tax, 2),
            'tax_type': tax_type,
            'effective_tax_rate': round(tax / gains * 100, 2) if gains > 0 else 0
        }


class GoalPlanner:
    """Goal-based investment planning"""
    
    @staticmethod
    def plan_retirement(current_age: int,
                       retirement_age: int,
                       monthly_expenses: float,
                       inflation_rate: float = 0.06,
                       post_retirement_years: int = 25,
                       expected_return: float = 0.12) -> Dict:
        """
        Calculate retirement corpus and SIP required
        """
        years_to_retirement = retirement_age - current_age
        
        # Future monthly expenses at retirement
        future_monthly_expenses = monthly_expenses * ((1 + inflation_rate) ** years_to_retirement)
        
        # Corpus needed (assuming 6% return post-retirement)
        post_ret_return = 0.06
        corpus_needed = future_monthly_expenses * 12 * (
            (1 - (1 + post_ret_return) ** -post_retirement_years) / post_ret_return
        )
        
        # Calculate required SIP
        required_sip = SIPCalculator.calculate_required_sip(
            corpus_needed, expected_return, years_to_retirement
        )
        
        return {
            'years_to_retirement': years_to_retirement,
            'current_monthly_expenses': round(monthly_expenses, 2),
            'future_monthly_expenses': round(future_monthly_expenses, 2),
            'corpus_needed': round(corpus_needed, 2),
            'required_monthly_sip': round(required_sip, 2),
            'post_retirement_years': post_retirement_years
        }
    
    @staticmethod
    def plan_child_education(years_until_needed: int,
                            current_cost: float,
                            inflation_rate: float = 0.10,
                            expected_return: float = 0.12) -> Dict:
        """
        Calculate child education planning
        """
        future_cost = current_cost * ((1 + inflation_rate) ** years_until_needed)
        
        required_sip = SIPCalculator.calculate_required_sip(
            future_cost, expected_return, years_until_needed
        )
        
        # Also calculate lumpsum option
        pv_factor = (1 + expected_return) ** years_until_needed
        required_lumpsum = future_cost / pv_factor
        
        return {
            'years_until_needed': years_until_needed,
            'current_cost': round(current_cost, 2),
            'future_cost': round(future_cost, 2),
            'required_monthly_sip': round(required_sip, 2),
            'required_lumpsum_today': round(required_lumpsum, 2),
            'total_sip_investment': round(required_sip * 12 * years_until_needed, 2)
        }


class PortfolioAnalyzer:
    """Analyze and optimize portfolio"""
    
    @staticmethod
    def calculate_portfolio_metrics(portfolio: pd.DataFrame) -> Dict:
        """
        Calculate comprehensive portfolio metrics
        portfolio DataFrame should have: allocation_percentage, returns_1yr, returns_3yr, sharpe, alpha, beta
        """
        # Weighted returns
        returns_1yr = (portfolio['allocation_percentage'] * portfolio['returns_1yr']).sum() / 100
        
        # Handle returns_3yr with NaN values
        returns_3yr_clean = portfolio[['allocation_percentage', 'returns_3yr']].copy()
        returns_3yr_clean = returns_3yr_clean.dropna(subset=['returns_3yr'])
        if len(returns_3yr_clean) > 0:
            returns_3yr = (returns_3yr_clean['allocation_percentage'] * returns_3yr_clean['returns_3yr']).sum() / returns_3yr_clean['allocation_percentage'].sum()
        else:
            returns_3yr = returns_1yr
        
        # Weighted risk metrics (handle NaN values)
        sharpe = (portfolio['allocation_percentage'] * portfolio['sharpe'].fillna(0)).sum() / 100
        alpha = (portfolio['allocation_percentage'] * portfolio['alpha'].fillna(0)).sum() / 100
        beta = (portfolio['allocation_percentage'] * portfolio['beta'].fillna(1)).sum() / 100
        
        # Diversification score (based on number of funds and category spread)
        num_funds = len(portfolio)
        num_categories = portfolio['category'].nunique()
        num_amcs = portfolio['amc_name'].nunique()
        
        diversification_score = min(
            (num_funds / 10 * 0.4 + num_categories / 3 * 0.3 + num_amcs / num_funds * 0.3) * 100,
            100
        )
        
        return {
            'expected_return_1yr': round(returns_1yr, 2),
            'expected_return_3yr': round(returns_3yr, 2),
            'portfolio_sharpe': round(sharpe, 2),
            'portfolio_alpha': round(alpha, 2),
            'portfolio_beta': round(beta, 2),
            'diversification_score': round(diversification_score, 2),
            'num_funds': num_funds,
            'num_categories': num_categories,
            'num_amcs': num_amcs,
            'avg_expense_ratio': round(
                (portfolio['allocation_percentage'] * portfolio['expense_ratio'].fillna(0)).sum() / 100, 2
            )
        }
    
    @staticmethod
    def check_rebalancing_needed(portfolio: pd.DataFrame,
                                 target_allocation: Dict[str, float]) -> Tuple[bool, List[Dict]]:
        """
        Check if portfolio needs rebalancing
        Returns: (needs_rebalancing, suggested_changes)
        """
        current_allocation = portfolio.groupby('category')['allocation_percentage'].sum().to_dict()
        
        suggestions = []
        needs_rebalancing = False
        threshold = 5  # 5% deviation threshold
        
        for category, target_pct in target_allocation.items():
            current_pct = current_allocation.get(category, 0)
            deviation = abs(current_pct - target_pct)
            
            if deviation > threshold:
                needs_rebalancing = True
                suggestions.append({
                    'category': category,
                    'current_allocation': round(current_pct, 2),
                    'target_allocation': round(target_pct, 2),
                    'deviation': round(deviation, 2),
                    'action': 'Increase' if current_pct < target_pct else 'Decrease',
                    'amount_change': round(deviation, 2)
                })
        
        return needs_rebalancing, suggestions
    
    @staticmethod
    def calculate_risk_return_ratio(portfolio: pd.DataFrame) -> float:
        """
        Calculate overall risk-return efficiency
        """
        metrics = PortfolioAnalyzer.calculate_portfolio_metrics(portfolio)
        
        # Risk-return ratio (higher is better)
        risk_score = (metrics['portfolio_beta'] + (100 - metrics['diversification_score']) / 100) / 2
        return_score = metrics['expected_return_3yr']
        
        if risk_score == 0:
            return 0
        
        return round(return_score / risk_score, 2)
    
    @staticmethod
    def analyze_with_ai_persona(portfolio: pd.DataFrame, 
                               age: int,
                               horizon: int,
                               risk_reaction: str,
                               goal: str,
                               volatility: str) -> Dict:
        """
        Analyze portfolio using AI investor persona engine
        
        Args:
            portfolio (pd.DataFrame): Portfolio to analyze
            age (int): Investor age
            horizon (int): Investment horizon in years
            risk_reaction (str): How investor reacts to market downturns
            goal (str): Primary financial goal
            volatility (str): Comfort level with portfolio fluctuations
            
        Returns:
            Dict: Analysis results including persona information and recommendations
        """
        if not AI_PERSONA_AVAILABLE:
            raise ImportError("AI Risk Persona Engine not available")
        
        # Get AI investor persona
        persona = get_investor_persona(age, horizon, risk_reaction, goal, volatility)
        
        # Calculate standard metrics
        metrics = PortfolioAnalyzer.calculate_portfolio_metrics(portfolio)
        
        # Check if allocation aligns with persona recommendations
        current_equity = metrics.get('expected_return_3yr', 50)  # Simplified for example
        suggested_equity_range = persona['allocation_range']['equity']
        
        allocation_aligned = (
            suggested_equity_range[0] <= current_equity <= suggested_equity_range[1]
        )
        
        return {
            'persona': persona,
            'metrics': metrics,
            'allocation_aligned': allocation_aligned,
            'recommendation': (
                "Your portfolio aligns well with your investor persona" 
                if allocation_aligned 
                else "Consider adjusting your portfolio to better match your investor persona"
            )
        }
