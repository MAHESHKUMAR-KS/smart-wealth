"""
Configuration file for WealthyWise Investment Platform
"""
import os

# File Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data.csv')

# Investment Constraints
MIN_INVESTMENT = 1000
MAX_INVESTMENT = 10000000
MIN_AGE = 18
MAX_AGE = 80
MIN_INVESTMENT_HORIZON = 1  # years
MAX_INVESTMENT_HORIZON = 40  # years

# Portfolio Rules
MIN_FUNDS_IN_PORTFOLIO = 3
MAX_FUNDS_IN_PORTFOLIO = 10
MAX_SINGLE_FUND_ALLOCATION = 0.25  # 25% max in one fund
MIN_DIVERSIFICATION_SCORE = 0.6

# Fund Quality Thresholds
QUALITY_THRESHOLDS = {
    'excellent': {
        'sharpe': 1.5,
        'sortino': 1.5,
        'alpha': 3.0,
        'rating': 4
    },
    'good': {
        'sharpe': 1.0,
        'sortino': 1.0,
        'alpha': 1.0,
        'rating': 3
    },
    'acceptable': {
        'sharpe': 0.5,
        'sortino': 0.5,
        'alpha': 0.0,
        'rating': 2
    }
}

# Risk Profiles
RISK_PROFILES = {
    'conservative': {
        'name': 'Conservative',
        'description': 'Focus on capital preservation with steady returns',
        'equity_range': (0, 30),
        'debt_range': (50, 80),
        'hybrid_range': (10, 30),
        'volatility_tolerance': 'low',
        'expected_return': 0.08
    },
    'moderate': {
        'name': 'Moderate',
        'description': 'Balanced growth with moderate risk',
        'equity_range': (30, 60),
        'debt_range': (20, 50),
        'hybrid_range': (10, 30),
        'volatility_tolerance': 'medium',
        'expected_return': 0.12
    },
    'aggressive': {
        'name': 'Aggressive',
        'description': 'Maximum growth potential with higher risk',
        'equity_range': (60, 100),
        'debt_range': (0, 20),
        'hybrid_range': (0, 20),
        'volatility_tolerance': 'high',
        'expected_return': 0.15
    }
}

# Tax Rates (India - as of FY 2024-25)
TAX_RATES = {
    'ltcg_equity': 0.10,  # Long term capital gains > 1 lakh
    'stcg_equity': 0.15,  # Short term capital gains
    'ltcg_debt': 0.20,    # Long term capital gains with indexation
    'stcg_debt': 0.30,    # Added to income tax slab
    'elss_deduction': 150000  # 80C limit
}

# SIP Settings
SIP_STEP_UP_OPTIONS = [5, 10, 15, 20]  # % per year
SIP_FREQUENCIES = ['Monthly', 'Quarterly', 'Yearly']

# Investment Goals
INVESTMENT_GOALS = {
    'retirement': {
        'name': 'Retirement Planning',
        'typical_horizon': 30,
        'suggested_profile': 'moderate',
        'inflation_rate': 0.06
    },
    'child_education': {
        'name': 'Child Education',
        'typical_horizon': 15,
        'suggested_profile': 'moderate',
        'inflation_rate': 0.10
    },
    'home_purchase': {
        'name': 'Home Purchase',
        'typical_horizon': 10,
        'suggested_profile': 'moderate',
        'inflation_rate': 0.08
    },
    'wealth_creation': {
        'name': 'Wealth Creation',
        'typical_horizon': 20,
        'suggested_profile': 'aggressive',
        'inflation_rate': 0.06
    },
    'emergency_fund': {
        'name': 'Emergency Fund',
        'typical_horizon': 1,
        'suggested_profile': 'conservative',
        'inflation_rate': 0.05
    }
}

# Disclaimer Text
DISCLAIMER = """
**Investment Disclaimer:**

This platform provides educational information and analysis tools only. It is NOT:
- Personalized investment advice from a SEBI registered advisor
- A guarantee of returns or fund performance
- A recommendation to buy or sell any security

**Important Points:**
- Past performance does not guarantee future results
- Mutual fund investments are subject to market risks
- Please read all scheme related documents carefully
- Consult a qualified financial advisor before investing
- Tax laws are subject to change

**Data Accuracy:**
Fund data is sourced from publicly available information and may not be real-time. 
Always verify current information from official fund house websites or registrars.
"""
