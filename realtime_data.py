"""
Real-time Data Fetcher and Simulator
Supports both simulated real-time data and API integration
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import random


class RealTimeDataFetcher:
    """Fetch and simulate real-time mutual fund data"""
    
    def __init__(self, base_data: pd.DataFrame):
        """Initialize with base fund data"""
        self.base_data = base_data.copy()
        self.last_update = datetime.now()
        
    def simulate_market_movement(self, volatility: float = 0.02) -> pd.DataFrame:
        """
        Simulate real-time market movements
        In production, this would be replaced with actual API calls
        """
        data = self.base_data.copy()
        
        # Simulate NAV changes (±2% daily movement)
        market_factor = np.random.normal(0, volatility)
        
        # Update returns based on simulated movement
        if 'returns_1yr' in data.columns:
            data['returns_1yr_updated'] = data['returns_1yr'] + (market_factor * 100)
            data['current_nav_change'] = market_factor * 100
            
        # Simulate volume changes
        data['today_volume'] = np.random.randint(1000, 100000, size=len(data))
        
        # Market sentiment (randomly assign)
        data['market_sentiment'] = np.random.choice(['Bullish', 'Neutral', 'Bearish'], size=len(data))
        
        self.last_update = datetime.now()
        data['last_updated'] = self.last_update
        
        return data
    
    def get_live_nav(self, scheme_name: str) -> Dict:
        """
        Get live NAV for a specific fund
        In production: Connect to MFCentral/AMFI API
        """
        fund = self.base_data[self.base_data['scheme_name'] == scheme_name]
        
        if len(fund) == 0:
            return None
        
        fund = fund.iloc[0]
        
        # Simulate live NAV
        base_nav = 100  # Simulated base NAV
        change_pct = np.random.normal(0, 0.01)  # ±1% daily change
        current_nav = base_nav * (1 + change_pct)
        
        return {
            'scheme_name': scheme_name,
            'current_nav': round(current_nav, 2),
            'previous_nav': base_nav,
            'change': round(change_pct * 100, 2),
            'change_amount': round(current_nav - base_nav, 2),
            'as_of_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'amc_name': fund['amc_name'],
            'category': fund['category']
        }
    
    def get_market_overview(self) -> Dict:
        """Get overall market statistics"""
        data = self.simulate_market_movement()
        
        return {
            'total_funds_tracked': len(data),
            'avg_return_today': round(data['current_nav_change'].mean(), 2),
            'positive_movers': len(data[data['current_nav_change'] > 0]),
            'negative_movers': len(data[data['current_nav_change'] < 0]),
            'neutral_movers': len(data[data['current_nav_change'] == 0]),
            'top_gainer': {
                'name': data.nlargest(1, 'current_nav_change').iloc[0]['scheme_name'],
                'change': round(data['current_nav_change'].max(), 2)
            },
            'top_loser': {
                'name': data.nsmallest(1, 'current_nav_change').iloc[0]['scheme_name'],
                'change': round(data['current_nav_change'].min(), 2)
            },
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def get_trending_funds(self, limit: int = 10) -> pd.DataFrame:
        """Get trending/most searched funds"""
        data = self.simulate_market_movement()
        
        # Simulate search volume
        data['search_volume'] = np.random.randint(100, 10000, size=len(data))
        
        trending = data.nlargest(limit, 'search_volume')[
            ['scheme_name', 'category', 'amc_name', 'returns_1yr', 
             'current_nav_change', 'search_volume', 'market_sentiment']
        ]
        
        return trending


class HistoricalDataSimulator:
    """Generate historical data for backtesting"""
    
    @staticmethod
    def generate_price_history(fund_data: pd.Series, years: int = 5) -> pd.DataFrame:
        """
        Generate simulated historical NAV data
        In production: Fetch from historical database/API
        """
        # Use fund's actual returns to estimate volatility
        annual_return = fund_data['returns_1yr'] / 100 if pd.notna(fund_data['returns_1yr']) else 0.12
        volatility = fund_data['sd'] / 100 if pd.notna(fund_data['sd']) else 0.15
        
        # Generate daily data points
        days = years * 252  # Trading days
        dates = pd.date_range(end=datetime.now(), periods=days, freq='B')
        
        # Geometric Brownian Motion for NAV simulation
        initial_nav = 100
        dt = 1/252
        
        # Generate random returns
        daily_returns = np.random.normal(
            annual_return * dt,
            volatility * np.sqrt(dt),
            days
        )
        
        # Calculate cumulative NAV
        nav_values = initial_nav * np.exp(np.cumsum(daily_returns))
        
        # Ensure we have the right shape
        if len(nav_values) != len(dates):
            min_len = min(len(nav_values), len(dates))
            nav_values = nav_values[:min_len]
            dates = dates[:min_len]
        
        history = pd.DataFrame({
            'date': dates,
            'nav': nav_values,
            'daily_return': daily_returns[:len(nav_values)] * 100 if len(daily_returns) >= len(nav_values) else np.zeros(len(nav_values))
        })
        
        return history
    
    @staticmethod
    def generate_benchmark_history(years: int = 5, benchmark_type: str = 'nifty50') -> pd.DataFrame:
        """Generate benchmark index history"""
        
        # Benchmark parameters
        benchmarks = {
            'nifty50': {'return': 0.12, 'volatility': 0.18},
            'sensex': {'return': 0.11, 'volatility': 0.17},
            'nifty_midcap': {'return': 0.15, 'volatility': 0.22},
            'nifty_smallcap': {'return': 0.18, 'volatility': 0.28}
        }
        
        params = benchmarks.get(benchmark_type, benchmarks['nifty50'])
        days = years * 252
        dates = pd.date_range(end=datetime.now(), periods=days, freq='B')
        
        initial_value = 10000
        dt = 1/252
        
        daily_returns = np.random.normal(
            params['return'] * dt,
            params['volatility'] * np.sqrt(dt),
            days
        )
        
        values = initial_value * np.exp(np.cumsum(daily_returns))
        
        return pd.DataFrame({
            'date': dates,
            'value': values,
            'daily_return': daily_returns * 100
        })


class LivePortfolioTracker:
    """Track portfolio performance in real-time"""
    
    def __init__(self, portfolio: pd.DataFrame, initial_investment: float):
        self.portfolio = portfolio
        self.initial_investment = initial_investment
        self.start_date = datetime.now()
        
    def calculate_current_value(self, data_fetcher: RealTimeDataFetcher) -> Dict:
        """Calculate current portfolio value with live NAV"""
        current_data = data_fetcher.simulate_market_movement()
        
        portfolio_value = 0
        fund_values = []
        
        for _, fund in self.portfolio.iterrows():
            # Match fund in current data
            current_fund = current_data[current_data['scheme_name'] == fund['scheme_name']]
            
            if len(current_fund) > 0:
                current_fund = current_fund.iloc[0]
                
                # Calculate current value
                invested = fund['investment_amount']
                change = current_fund.get('current_nav_change', 0) / 100
                current_val = invested * (1 + change)
                
                portfolio_value += current_val
                
                fund_values.append({
                    'scheme_name': fund['scheme_name'],
                    'invested': invested,
                    'current_value': round(current_val, 2),
                    'gain_loss': round(current_val - invested, 2),
                    'gain_loss_pct': round(change * 100, 2)
                })
        
        total_gain = portfolio_value - self.initial_investment
        
        return {
            'portfolio_value': round(portfolio_value, 2),
            'total_invested': self.initial_investment,
            'total_gain': round(total_gain, 2),
            'total_return_pct': round((total_gain / self.initial_investment) * 100, 2),
            'funds': fund_values,
            'as_of': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def get_performance_alerts(self, threshold: float = 5.0) -> List[Dict]:
        """Get alerts for significant fund movements"""
        alerts = []
        
        # This would check against real-time data
        # For now, simulate some alerts
        if random.random() > 0.7:
            alerts.append({
                'type': 'warning',
                'fund': self.portfolio.iloc[0]['scheme_name'],
                'message': f'Fund dropped by {threshold}% today',
                'timestamp': datetime.now()
            })
        
        return alerts


class NewsSimulator:
    """Simulate market news and fund updates"""
    
    @staticmethod
    def get_latest_news(category: str = None, limit: int = 5) -> List[Dict]:
        """Generate simulated market news"""
        news_templates = [
            "Top {category} funds show strong performance in Q4",
            "Market volatility affects {category} sector investments",
            "New regulations impact {category} mutual funds",
            "Expert recommends {category} funds for long-term growth",
            "AMC announces dividend for select {category} schemes"
        ]
        
        categories = ['Equity', 'Debt', 'Hybrid', 'Index', 'Sectoral']
        
        news = []
        for i in range(limit):
            cat = category if category else random.choice(categories)
            news.append({
                'headline': random.choice(news_templates).format(category=cat),
                'category': cat,
                'timestamp': datetime.now() - timedelta(hours=random.randint(1, 24)),
                'source': random.choice(['Economic Times', 'MoneyControl', 'LiveMint', 'Value Research']),
                'sentiment': random.choice(['Positive', 'Neutral', 'Negative'])
            })
        
        return sorted(news, key=lambda x: x['timestamp'], reverse=True)
