"""
Backtesting Engine and Performance Comparison Tools
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from realtime_data import HistoricalDataSimulator


class BacktestEngine:
    """Backtest investment strategies and portfolios"""
    
    def __init__(self, fund_data: pd.DataFrame):
        self.fund_data = fund_data
        self.simulator = HistoricalDataSimulator()
        
    def backtest_portfolio(self, 
                          portfolio: pd.DataFrame,
                          years: int = 5,
                          rebalance_frequency: str = 'yearly') -> Dict:
        """
        Backtest a portfolio over historical period
        """
        results = {
            'portfolio_performance': pd.DataFrame(),
            'fund_performances': {},
            'rebalance_events': [],
            'metrics': {}
        }
        
        # Check if portfolio is empty
        if len(portfolio) == 0:
            return results
        
        # Generate historical data for each fund
        fund_histories = []
        
        for _, fund in portfolio.iterrows():
            # Find fund in base data
            fund_match = self.fund_data[
                self.fund_data['scheme_name'] == fund['scheme_name']
            ]
            
            if len(fund_match) > 0:
                fund_series = fund_match.iloc[0]
                
                # Generate history
                try:
                    history = self.simulator.generate_price_history(fund_series, years)
                    
                    # Calculate returns
                    initial_investment = fund['investment_amount']
                    if len(history) > 0 and history.iloc[0]['nav'] > 0:
                        shares = initial_investment / history.iloc[0]['nav']
                        history['portfolio_value'] = shares * history['nav']
                        history['scheme_name'] = fund['scheme_name']
                        fund_histories.append(history)
                except Exception as e:
                    print(f"Warning: Could not generate history for {fund['scheme_name']}: {e}")
                    continue
        
        # Check if we have any fund histories
        if not fund_histories:
            return results
        
        # Combine portfolio performance
        # Start with the first fund's dates
        combined_portfolio = fund_histories[0][['date']].copy()
        
        # Add each fund's portfolio value
        for history in fund_histories:
            combined_portfolio = combined_portfolio.merge(
                history[['date', 'portfolio_value', 'nav']],
                on='date',
                how='outer',
                suffixes=('', f"_{history['scheme_name'][:10]}")
            )
        
        # Fill NaN values and calculate total portfolio value
        value_columns = [col for col in combined_portfolio.columns if col.startswith('portfolio_value')]
        combined_portfolio[value_columns] = combined_portfolio[value_columns].fillna(0)
        combined_portfolio['total_value'] = combined_portfolio[value_columns].sum(axis=1)
        
        # Calculate daily returns
        combined_portfolio['daily_return'] = combined_portfolio['total_value'].pct_change() * 100
        
        results['portfolio_performance'] = combined_portfolio
        
        # Store individual fund performances
        for history in fund_histories:
            scheme_name = history['scheme_name'].iloc[0] if isinstance(history['scheme_name'], pd.Series) else history['scheme_name']
            results['fund_performances'][scheme_name] = history
        
        # Calculate metrics if we have data
        if len(combined_portfolio) > 1 and combined_portfolio.iloc[0]['total_value'] > 0:
            initial_value = combined_portfolio.iloc[0]['total_value']
            final_value = combined_portfolio.iloc[-1]['total_value']
            
            total_return = ((final_value - initial_value) / initial_value) * 100
            annualized_return = ((final_value / initial_value) ** (1/years) - 1) * 100 if years > 0 else 0
            
            # Risk metrics
            daily_returns = combined_portfolio['daily_return'].dropna()
            if len(daily_returns) > 1:
                volatility = daily_returns.std() * np.sqrt(252)  # Annualized
                sharpe_ratio = (annualized_return - 6) / volatility if volatility > 0 else 0  # Assuming 6% risk-free rate
                
                max_value = combined_portfolio['total_value'].expanding().max()
                drawdown = (combined_portfolio['total_value'] - max_value) / max_value * 100
                max_drawdown = drawdown.min() if len(drawdown) > 0 else 0
            else:
                volatility = 0
                sharpe_ratio = 0
                max_drawdown = 0
            
            results['metrics'] = {
                'initial_value': round(initial_value, 2),
                'final_value': round(final_value, 2),
                'total_return': round(total_return, 2),
                'annualized_return': round(annualized_return, 2),
                'volatility': round(volatility, 2),
                'sharpe_ratio': round(sharpe_ratio, 2),
                'max_drawdown': round(max_drawdown, 2),
                'best_day': round(daily_returns.max(), 2) if len(daily_returns) > 0 else 0,
                'worst_day': round(daily_returns.min(), 2) if len(daily_returns) > 0 else 0,
                'positive_days': len(daily_returns[daily_returns > 0]) if len(daily_returns) > 0 else 0,
                'negative_days': len(daily_returns[daily_returns < 0]) if len(daily_returns) > 0 else 0
            }
        
        return results
    
    def compare_with_benchmark(self,
                               portfolio: pd.DataFrame,
                               benchmark: str = 'nifty50',
                               years: int = 5) -> Dict:
        """Compare portfolio performance against benchmark"""
        
        # Backtest portfolio
        portfolio_results = self.backtest_portfolio(portfolio, years)
        
        # Check if we have results
        if len(portfolio_results['portfolio_performance']) == 0:
            return {
                'comparison_chart': pd.DataFrame(),
                'portfolio_metrics': {},
                'benchmark_return': 0,
                'alpha': 0,
                'outperformance': False,
                'tracking_error': 0
            }
        
        # Generate benchmark data
        benchmark_history = self.simulator.generate_benchmark_history(years, benchmark)
        
        # Align dates
        portfolio_perf = portfolio_results['portfolio_performance']
        
        # Merge for comparison
        comparison = pd.DataFrame({
            'date': portfolio_perf['date'],
            'portfolio_value': portfolio_perf['total_value']
        })
        
        # Add benchmark values (align dates)
        benchmark_aligned = pd.merge(
            comparison[['date']], 
            benchmark_history[['date', 'value']], 
            on='date', 
            how='left'
        )
        
        comparison['benchmark_value'] = benchmark_aligned['value'].fillna(method='ffill').fillna(method='bfill')
        
        # Normalize to 100 for easy comparison
        if len(comparison) > 0 and comparison.iloc[0]['portfolio_value'] > 0:
            comparison['portfolio_normalized'] = (
                comparison['portfolio_value'] / comparison.iloc[0]['portfolio_value'] * 100
            )
        else:
            comparison['portfolio_normalized'] = 100
            
        if len(comparison) > 0 and comparison.iloc[0]['benchmark_value'] > 0:
            comparison['benchmark_normalized'] = (
                comparison['benchmark_value'] / comparison.iloc[0]['benchmark_value'] * 100
            )
        else:
            comparison['benchmark_normalized'] = 100
        
        # Calculate alpha (excess return over benchmark)
        alpha = 0
        benchmark_return = 0
        outperformance = False
        tracking_error = 0
        
        if len(portfolio_results['metrics']) > 0 and len(comparison) > 1:
            portfolio_return = portfolio_results['metrics'].get('annualized_return', 0)
            
            # Calculate benchmark return
            if comparison.iloc[0]['benchmark_value'] > 0 and years > 0:
                benchmark_return = (
                    (comparison.iloc[-1]['benchmark_value'] / comparison.iloc[0]['benchmark_value']) ** (1/years) - 1
                ) * 100
            
            alpha = portfolio_return - benchmark_return
            outperformance = alpha > 0
            
            # Tracking error
            if len(comparison) > 1:
                tracking_error = round(
                    (comparison['portfolio_normalized'] - comparison['benchmark_normalized']).std(), 2
                )
        
        return {
            'comparison_chart': comparison,
            'portfolio_metrics': portfolio_results['metrics'],
            'benchmark_return': round(benchmark_return, 2),
            'alpha': round(alpha, 2),
            'outperformance': outperformance,
            'tracking_error': tracking_error
        }
    
    def stress_test(self, portfolio: pd.DataFrame, scenarios: List[str] = None) -> Dict:
        """Test portfolio under different market scenarios"""
        
        if scenarios is None:
            scenarios = ['market_crash', 'high_inflation', 'recession', 'bull_market']
        
        scenario_params = {
            'market_crash': {'return': -0.30, 'volatility': 0.40},
            'high_inflation': {'return': -0.10, 'volatility': 0.25},
            'recession': {'return': -0.15, 'volatility': 0.30},
            'bull_market': {'return': 0.25, 'volatility': 0.20}
        }
        
        results = {}
        
        for scenario in scenarios:
            params = scenario_params.get(scenario, scenario_params['market_crash'])
            
            # Simulate scenario impact
            initial_value = portfolio['investment_amount'].sum()
            
            # Apply scenario return
            scenario_value = initial_value * (1 + params['return'])
            loss_pct = ((scenario_value - initial_value) / initial_value) * 100
            
            # Estimate recovery time (simplified)
            if loss_pct < 0:
                recovery_years = abs(loss_pct) / 12  # Assuming 12% annual recovery
            else:
                recovery_years = 0
            
            results[scenario] = {
                'scenario_name': scenario.replace('_', ' ').title(),
                'projected_value': round(scenario_value, 2),
                'loss_amount': round(scenario_value - initial_value, 2),
                'loss_percentage': round(loss_pct, 2),
                'estimated_recovery_years': round(recovery_years, 1),
                'risk_level': 'High' if abs(loss_pct) > 20 else 'Medium' if abs(loss_pct) > 10 else 'Low'
            }
        
        return results


class PerformanceComparator:
    """Advanced fund and portfolio comparison"""
    
    @staticmethod
    def compare_funds_detailed(funds: List[pd.Series]) -> pd.DataFrame:
        """Detailed side-by-side fund comparison"""
        
        comparison_data = []
        
        for fund in funds:
            comparison_data.append({
                'Fund Name': fund['scheme_name'],
                'Category': fund['category'],
                'AMC': fund['amc_name'],
                'Quality Score': fund.get('quality_score', 0),
                'Risk Score': fund.get('risk_score', 0),
                '1Y Return': f"{fund['returns_1yr']:.2f}%",
                '3Y Return': f"{fund['returns_3yr']:.2f}%" if pd.notna(fund['returns_3yr']) else 'N/A',
                '5Y Return': f"{fund['returns_5yr']:.2f}%" if pd.notna(fund['returns_5yr']) else 'N/A',
                'Sharpe Ratio': f"{fund['sharpe']:.2f}",
                'Alpha': f"{fund['alpha']:.2f}",
                'Beta': f"{fund['beta']:.2f}" if pd.notna(fund['beta']) else 'N/A',
                'Expense Ratio': f"{fund['expense_ratio']:.2f}%",
                'Rating': '⭐' * int(fund['rating']) if pd.notna(fund['rating']) else 'N/A',
                'Fund Manager': fund['fund_manager']
            })
        
        return pd.DataFrame(comparison_data)
    
    @staticmethod
    def rank_funds(funds_df: pd.DataFrame, criteria: Dict[str, float]) -> pd.DataFrame:
        """
        Rank funds based on weighted criteria
        criteria: {'returns_3yr': 0.4, 'sharpe': 0.3, 'alpha': 0.3}
        """
        
        funds_copy = funds_df.copy()
        
        # Normalize each criterion to 0-1 scale
        normalized_scores = pd.DataFrame()
        
        for criterion, weight in criteria.items():
            if criterion in funds_copy.columns:
                col_data = pd.to_numeric(funds_copy[criterion], errors='coerce')
                min_val = col_data.min()
                max_val = col_data.max()
                
                if max_val > min_val:
                    normalized = (col_data - min_val) / (max_val - min_val)
                    normalized_scores[criterion] = normalized * weight
        
        # Calculate total weighted score
        funds_copy['composite_score'] = normalized_scores.sum(axis=1) * 100
        
        # Rank
        funds_copy['rank'] = funds_copy['composite_score'].rank(ascending=False, method='min')
        
        return funds_copy.sort_values('rank')
    
    @staticmethod
    def peer_comparison(fund: pd.Series, fund_universe: pd.DataFrame) -> Dict:
        """Compare fund against category peers"""
        
        # Filter peers in same category
        peers = fund_universe[fund_universe['category'] == fund['category']]
        
        if len(peers) == 0:
            return None
        
        # Calculate percentiles
        returns_percentile = (peers['returns_3yr'] < fund['returns_3yr']).sum() / len(peers) * 100
        sharpe_percentile = (peers['sharpe'] < fund['sharpe']).sum() / len(peers) * 100
        expense_percentile = (peers['expense_ratio'] > fund['expense_ratio']).sum() / len(peers) * 100
        
        return {
            'category': fund['category'],
            'total_peers': len(peers),
            'returns_percentile': round(returns_percentile, 1),
            'sharpe_percentile': round(sharpe_percentile, 1),
            'expense_percentile': round(expense_percentile, 1),
            'avg_peer_return': round(peers['returns_3yr'].mean(), 2),
            'avg_peer_sharpe': round(peers['sharpe'].mean(), 2),
            'better_than_peers': returns_percentile > 50 and sharpe_percentile > 50,
            'rank_in_category': int((peers['returns_3yr'] > fund['returns_3yr']).sum() + 1)
        }


class WhatIfAnalyzer:
    """What-if scenario analysis for investment planning"""
    
    @staticmethod
    def analyze_sip_variations(base_sip: float,
                               years: int,
                               expected_return: float,
                               variations: Dict[str, float] = None) -> Dict:
        """
        Analyze different SIP scenarios
        variations: {'pessimistic': 0.08, 'optimistic': 0.15}
        """
        
        if variations is None:
            variations = {
                'Pessimistic (8%)': 0.08,
                'Base Case (12%)': expected_return,
                'Optimistic (15%)': 0.15,
                'Very Optimistic (18%)': 0.18
            }
        
        results = {}
        
        for scenario_name, annual_return in variations.items():
            monthly_rate = annual_return / 12
            months = years * 12
            
            # Future value of SIP
            future_value = base_sip * (
                ((1 + monthly_rate) ** months - 1) / monthly_rate
            ) * (1 + monthly_rate)
            
            total_invested = base_sip * months
            gains = future_value - total_invested
            
            results[scenario_name] = {
                'future_value': round(future_value, 2),
                'total_invested': round(total_invested, 2),
                'gains': round(gains, 2),
                'return_pct': round((gains / total_invested) * 100, 2)
            }
        
        return results
    
    @staticmethod
    def goal_sensitivity_analysis(target_amount: float,
                                  years: int,
                                  expected_return: float = 0.12) -> Dict:
        """Analyze sensitivity to different return rates"""
        
        return_scenarios = np.arange(0.06, 0.20, 0.02)  # 6% to 18%
        
        results = []
        
        for annual_return in return_scenarios:
            monthly_rate = annual_return / 12
            months = years * 12
            
            # Calculate required SIP
            required_sip = target_amount * monthly_rate / (
                ((1 + monthly_rate) ** months - 1) * (1 + monthly_rate)
            )
            
            results.append({
                'return_rate': round(annual_return * 100, 1),
                'required_monthly_sip': round(required_sip, 2),
                'total_investment': round(required_sip * months, 2),
                'gains': round(target_amount - (required_sip * months), 2)
            })
        
        return {
            'scenarios': results,
            'recommendation': 'Start with higher SIP to account for market volatility'
        }
    
    @staticmethod
    def portfolio_reallocation_impact(current_portfolio: pd.DataFrame,
                                     new_allocation: Dict[str, float]) -> Dict:
        """
        Analyze impact of changing portfolio allocation
        new_allocation: {'Equity': 70, 'Debt': 20, 'Hybrid': 10}
        """
        
        current_allocation = current_portfolio.groupby('category')['allocation_percentage'].sum().to_dict()
        
        total_value = current_portfolio['investment_amount'].sum()
        
        # Calculate expected returns for each allocation
        def calc_expected_return(allocation):
            # Rough estimates based on category
            category_returns = {'Equity': 0.12, 'Debt': 0.07, 'Hybrid': 0.10}
            return sum(allocation.get(cat, 0) / 100 * ret for cat, ret in category_returns.items())
        
        current_return = calc_expected_return(current_allocation)
        new_return = calc_expected_return(new_allocation)
        
        # Project 5-year impact
        years = 5
        current_fv = total_value * ((1 + current_return) ** years)
        new_fv = total_value * ((1 + new_return) ** years)
        
        return {
            'current_allocation': current_allocation,
            'new_allocation': new_allocation,
            'current_expected_return': round(current_return * 100, 2),
            'new_expected_return': round(new_return * 100, 2),
            'return_difference': round((new_return - current_return) * 100, 2),
            'projected_value_5yr_current': round(current_fv, 2),
            'projected_value_5yr_new': round(new_fv, 2),
            'potential_gain': round(new_fv - current_fv, 2),
            'recommendation': 'Beneficial' if new_fv > current_fv else 'Not recommended'
        }
