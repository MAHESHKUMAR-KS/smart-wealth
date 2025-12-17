"""
WealthyWise Advanced - Real-time Investment Platform
Includes: Real-time data, Backtesting, What-if scenarios, Performance comparison
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime

from fund_analyzer import FundAnalyzer
from portfolio_tools import SIPCalculator, GoalPlanner, PortfolioAnalyzer
from realtime_data import RealTimeDataFetcher, LivePortfolioTracker, NewsSimulator
from backtesting import BacktestEngine, PerformanceComparator, WhatIfAnalyzer
from config import *

# Page configuration
st.set_page_config(
    page_title="WealthyWise Advanced",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with modern theme
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(120deg, #1e3a8a, #7c3aed, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
        animation: gradient 3s ease infinite;
    }
    .live-badge {
        display: inline-block;
        background: linear-gradient(90deg, #ef4444, #dc2626);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.875rem;
        font-weight: 600;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .news-card {
        border-left: 4px solid #7c3aed;
        padding: 1rem;
        margin: 0.5rem 0;
        background: #f3f4f6;
        border-radius: 4px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border: 1px solid #e5e7eb;
        color: #1f2937;
    }
    .positive { color: #059669; font-weight: bold; }
    .negative { color: #dc2626; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = FundAnalyzer()
if 'realtime_fetcher' not in st.session_state:
    st.session_state.realtime_fetcher = RealTimeDataFetcher(st.session_state.analyzer.df)
if 'backtest_engine' not in st.session_state:
    st.session_state.backtest_engine = BacktestEngine(st.session_state.analyzer.df)
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = {}
if 'recommended_portfolio' not in st.session_state:
    st.session_state.recommended_portfolio = None
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now()


def show_live_market_dashboard():
    """Real-time market overview dashboard"""
    st.header("📊 Live Market Dashboard")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.markdown('<span class="live-badge">🔴 LIVE</span>', unsafe_allow_html=True)
        st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
        
        if st.button("🔄 Refresh Data"):
            st.session_state.last_refresh = datetime.now()
            st.rerun()
    
    # Get market overview
    market_data = st.session_state.realtime_fetcher.get_market_overview()
    
    # Metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Funds", market_data['total_funds_tracked'])
    
    with col2:
        change_class = "positive" if market_data['avg_return_today'] > 0 else "negative"
        st.markdown(f"**Avg. Return**")
        st.markdown(f"<p class='{change_class}'>{market_data['avg_return_today']:+.2f}%</p>", 
                   unsafe_allow_html=True)
    
    with col3:
        st.metric("Gainers", market_data['positive_movers'], 
                 delta=f"{market_data['positive_movers']}/{market_data['total_funds_tracked']}")
    
    with col4:
        st.metric("Losers", market_data['negative_movers'],
                 delta=f"-{market_data['negative_movers']}/{market_data['total_funds_tracked']}")
    
    with col5:
        st.metric("Neutral", market_data['neutral_movers'])
    
    # Top movers
    col1, col2 = st.columns(2)
    
    with col1:
        st.success(f"**🔥 Top Gainer:** {market_data['top_gainer']['name']}")
        st.markdown(f"<p class='positive'>+{market_data['top_gainer']['change']:.2f}%</p>", 
                   unsafe_allow_html=True)
    
    with col2:
        st.error(f"**📉 Top Loser:** {market_data['top_loser']['name']}")
        st.markdown(f"<p class='negative'>{market_data['top_loser']['change']:.2f}%</p>", 
                   unsafe_allow_html=True)
    
    # Trending funds
    st.subheader("🔥 Trending Funds")
    trending = st.session_state.realtime_fetcher.get_trending_funds(10)
    
    st.dataframe(
        trending.style.background_gradient(subset=['current_nav_change'], cmap='RdYlGn'),
        use_container_width=True,
        height=300
    )
    
    # Market news
    st.subheader("📰 Latest Market News")
    news = NewsSimulator.get_latest_news(limit=5)
    
    for item in news:
        sentiment_color = {
            'Positive': '🟢',
            'Neutral': '🟡',
            'Negative': '🔴'
        }.get(item['sentiment'], '🟡')
        
        st.markdown(f"""
        <div class="news-card">
            <strong>{sentiment_color} {item['headline']}</strong><br>
            <small>{item['source']} | {item['timestamp'].strftime('%H:%M')} | {item['category']}</small>
        </div>
        """, unsafe_allow_html=True)


def show_backtesting_lab():
    """Backtesting and performance analysis"""
    st.header("🔬 Backtesting Laboratory")
    
    if st.session_state.recommended_portfolio is None:
        st.warning("⚠️ Please generate a portfolio first from the Risk Profile section.")
        return
    
    portfolio = st.session_state.recommended_portfolio
    profile = st.session_state.user_profile
    
    tab1, tab2, tab3 = st.tabs(["📈 Historical Backtest", "⚖️ Benchmark Comparison", "⚠️ Stress Testing"])
    
    with tab1:
        st.subheader("Portfolio Historical Performance")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            years = st.slider("Backtest Period (Years)", 1, 10, 5)
            
            if st.button("🚀 Run Backtest", type="primary"):
                with st.spinner("Running historical simulation..."):
                    backtest_results = st.session_state.backtest_engine.backtest_portfolio(
                        portfolio, years
                    )
                    
                    st.session_state.backtest_results = backtest_results
        
        if 'backtest_results' in st.session_state:
            results = st.session_state.backtest_results
            metrics = results['metrics']
            
            # Metrics display
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Return", f"{metrics['total_return']:.1f}%")
            with col2:
                st.metric("Annual Return", f"{metrics['annualized_return']:.1f}%")
            with col3:
                st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
            with col4:
                st.metric("Max Drawdown", f"{metrics['max_drawdown']:.1f}%")
            
            # Performance chart
            perf_data = results['portfolio_performance']
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=perf_data['date'],
                y=perf_data['total_value'],
                mode='lines',
                name='Portfolio Value',
                fill='tozeroy',
                line=dict(color='rgb(124, 58, 237)', width=2)
            ))
            
            fig.update_layout(
                title='Portfolio Growth Over Time',
                xaxis_title='Date',
                yaxis_title='Portfolio Value (₹)',
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Additional stats
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Best Day", f"+{metrics['best_day']:.2f}%")
            with col2:
                st.metric("Worst Day", f"{metrics['worst_day']:.2f}%")
            with col3:
                win_rate = (metrics['positive_days'] / (metrics['positive_days'] + metrics['negative_days'])) * 100
                st.metric("Win Rate", f"{win_rate:.1f}%")
    
    with tab2:
        st.subheader("Benchmark Comparison")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            benchmark = st.selectbox(
                "Select Benchmark",
                ['nifty50', 'sensex', 'nifty_midcap', 'nifty_smallcap']
            )
            comp_years = st.slider("Comparison Period", 1, 10, 5, key="comp_years")
            
            if st.button("📊 Compare"):
                with st.spinner("Comparing with benchmark..."):
                    comparison = st.session_state.backtest_engine.compare_with_benchmark(
                        portfolio, benchmark, comp_years
                    )
                    st.session_state.benchmark_comparison = comparison
        
        if 'benchmark_comparison' in st.session_state:
            comp = st.session_state.benchmark_comparison
            
            # Alpha display
            if comp['alpha'] > 0:
                st.success(f"🎉 Portfolio outperformed {benchmark.upper()} by {comp['alpha']:.2f}% annually!")
            else:
                st.warning(f"Portfolio underperformed {benchmark.upper()} by {abs(comp['alpha']):.2f}% annually")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Portfolio Return", f"{comp['portfolio_metrics']['annualized_return']:.2f}%")
            with col2:
                st.metric("Benchmark Return", f"{comp['benchmark_return']:.2f}%")
            with col3:
                st.metric("Alpha", f"{comp['alpha']:+.2f}%")
            
            # Comparison chart
            chart_data = comp['comparison_chart']
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=chart_data['date'],
                y=chart_data['portfolio_normalized'],
                mode='lines',
                name='Your Portfolio',
                line=dict(color='rgb(124, 58, 237)', width=3)
            ))
            
            fig.add_trace(go.Scatter(
                x=chart_data['date'],
                y=chart_data['benchmark_normalized'],
                mode='lines',
                name=benchmark.upper(),
                line=dict(color='rgb(236, 72, 153)', width=2, dash='dash')
            ))
            
            fig.update_layout(
                title=f'Portfolio vs {benchmark.upper()} (Normalized to 100)',
                xaxis_title='Date',
                yaxis_title='Indexed Value',
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Stress Test Your Portfolio")
        
        st.info("💡 See how your portfolio performs under different market scenarios")
        
        if st.button("⚡ Run Stress Tests"):
            with st.spinner("Simulating market scenarios..."):
                stress_results = st.session_state.backtest_engine.stress_test(portfolio)
                
                for scenario_name, result in stress_results.items():
                    with st.expander(f"📊 {result['scenario_name']}", expanded=True):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                "Projected Value",
                                f"₹{result['projected_value']:,.0f}",
                                delta=f"{result['loss_percentage']:.1f}%"
                            )
                        
                        with col2:
                            loss_color = "🔴" if result['loss_percentage'] < 0 else "🟢"
                            st.metric(
                                f"{loss_color} Impact",
                                f"₹{abs(result['loss_amount']):,.0f}",
                            )
                        
                        with col3:
                            st.metric(
                                "Recovery Time",
                                f"{result['estimated_recovery_years']} years",
                                delta=result['risk_level']
                            )


def show_what_if_scenarios():
    """What-if scenario analysis"""
    st.header("🎯 What-If Scenario Analyzer")
    
    tab1, tab2, tab3 = st.tabs(["💰 SIP Variations", "🎯 Goal Sensitivity", "🔄 Reallocation Impact"])
    
    with tab1:
        st.subheader("SIP Return Scenarios")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            sip_amount = st.number_input("Monthly SIP (₹)", 1000, 100000, 10000, 1000)
            sip_years = st.slider("Investment Period", 5, 30, 15)
            
            if st.button("📊 Analyze Scenarios"):
                scenarios = WhatIfAnalyzer.analyze_sip_variations(
                    sip_amount, sip_years, 0.12
                )
                
                st.session_state.sip_scenarios = scenarios
        
        if 'sip_scenarios' in st.session_state:
            scenarios = st.session_state.sip_scenarios
            
            # Create comparison table
            scenario_df = pd.DataFrame(scenarios).T
            scenario_df.index.name = 'Scenario'
            
            st.dataframe(
                scenario_df.style.format({
                    'future_value': '₹{:,.0f}',
                    'total_invested': '₹{:,.0f}',
                    'gains': '₹{:,.0f}',
                    'return_pct': '{:.1f}%'
                }),
                use_container_width=True
            )
            
            # Visualization
            fig = go.Figure()
            
            for scenario_name, data in scenarios.items():
                fig.add_trace(go.Bar(
                    name=scenario_name,
                    x=['Invested', 'Gains'],
                    y=[data['total_invested'], data['gains']]
                ))
            
            fig.update_layout(
                title='Scenario Comparison',
                barmode='group',
                yaxis_title='Amount (₹)',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Goal Achievement Sensitivity")
        
        col1, col2 = st.columns(2)
        
        with col1:
            target = st.number_input("Target Amount (₹)", 100000, 50000000, 5000000, 100000)
            target_years = st.slider("Time Horizon (Years)", 5, 30, 15, key="target_years")
        
        if st.button("🎯 Analyze Sensitivity"):
            analysis = WhatIfAnalyzer.goal_sensitivity_analysis(target, target_years)
            
            st.session_state.sensitivity_analysis = analysis
        
        if 'sensitivity_analysis' in st.session_state:
            analysis = st.session_state.sensitivity_analysis
            
            df = pd.DataFrame(analysis['scenarios'])
            
            # Chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df['return_rate'],
                y=df['required_monthly_sip'],
                mode='lines+markers',
                name='Required SIP',
                line=dict(color='rgb(124, 58, 237)', width=3),
                fill='tozeroy'
            ))
            
            fig.update_layout(
                title='SIP Required vs Expected Return Rate',
                xaxis_title='Annual Return Rate (%)',
                yaxis_title='Monthly SIP Required (₹)',
                hovermode='x',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.info(f"💡 {analysis['recommendation']}")
            
            # Data table
            st.dataframe(
                df.style.format({
                    'return_rate': '{:.1f}%',
                    'required_monthly_sip': '₹{:,.0f}',
                    'total_investment': '₹{:,.0f}',
                    'gains': '₹{:,.0f}'
                }),
                use_container_width=True
            )
    
    with tab3:
        st.subheader("Portfolio Reallocation Impact")
        
        if st.session_state.recommended_portfolio is None:
            st.warning("Please generate a portfolio first")
            return
        
        portfolio = st.session_state.recommended_portfolio
        
        st.write("**Current Allocation:**")
        current_alloc = portfolio.groupby('category')['allocation_percentage'].sum().to_dict()
        
        col1, col2, col3 = st.columns(3)
        
        for i, (cat, pct) in enumerate(current_alloc.items()):
            with [col1, col2, col3][i % 3]:
                st.metric(cat, f"{pct:.1f}%")
        
        st.write("**Try New Allocation:**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            new_equity = st.slider("Equity %", 0, 100, int(current_alloc.get('Equity', 50)))
        with col2:
            new_debt = st.slider("Debt %", 0, 100, int(current_alloc.get('Debt', 30)))
        with col3:
            new_hybrid = st.slider("Hybrid %", 0, 100, int(current_alloc.get('Hybrid', 20)))
        
        total_new = new_equity + new_debt + new_hybrid
        
        if total_new != 100:
            st.error(f"⚠️ Total must equal 100%. Currently: {total_new}%")
        else:
            if st.button("📊 Analyze Impact"):
                new_allocation = {
                    'Equity': new_equity,
                    'Debt': new_debt,
                    'Hybrid': new_hybrid
                }
                
                impact = WhatIfAnalyzer.portfolio_reallocation_impact(
                    portfolio, new_allocation
                )
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        "Current Expected Return",
                        f"{impact['current_expected_return']:.2f}%"
                    )
                    st.metric(
                        "5-Year Projected Value",
                        f"₹{impact['projected_value_5yr_current']:,.0f}"
                    )
                
                with col2:
                    st.metric(
                        "New Expected Return",
                        f"{impact['new_expected_return']:.2f}%",
                        delta=f"{impact['return_difference']:+.2f}%"
                    )
                    st.metric(
                        "5-Year Projected Value",
                        f"₹{impact['projected_value_5yr_new']:,.0f}",
                        delta=f"₹{impact['potential_gain']:+,.0f}"
                    )
                
                if impact['recommendation'] == 'Beneficial':
                    st.success(f"✅ {impact['recommendation']} - This reallocation could increase returns!")
                else:
                    st.warning(f"⚠️ {impact['recommendation']} - Consider keeping current allocation")


def show_performance_comparison():
    """Advanced fund comparison tools"""
    st.header("⚖️ Advanced Performance Comparison")
    
    analyzer = st.session_state.analyzer
    
    st.subheader("Select Funds to Compare (3-5 funds recommended)")
    
    # Fund selection
    all_funds = analyzer.df['scheme_name'].tolist()
    selected_funds = st.multiselect(
        "Choose funds",
        all_funds,
        max_selections=5
    )
    
    if len(selected_funds) >= 2:
        # Get fund data
        funds_data = []
        for fund_name in selected_funds:
            fund = analyzer.df[analyzer.df['scheme_name'] == fund_name].iloc[0]
            funds_data.append(fund)
        
        # Detailed comparison
        comparison_df = PerformanceComparator.compare_funds_detailed(funds_data)
        
        st.dataframe(comparison_df, use_container_width=True, height=300)
        
        # Visual comparison
        col1, col2 = st.columns(2)
        
        with col1:
            # Returns comparison
            returns_fig = go.Figure()
            
            for fund in funds_data:
                returns_fig.add_trace(go.Bar(
                    name=fund['scheme_name'][:30],
                    x=['1Y', '3Y', '5Y'],
                    y=[
                        fund['returns_1yr'],
                        fund['returns_3yr'] if pd.notna(fund['returns_3yr']) else 0,
                        fund['returns_5yr'] if pd.notna(fund['returns_5yr']) else 0
                    ]
                ))
            
            returns_fig.update_layout(
                title='Returns Comparison',
                barmode='group',
                yaxis_title='Return (%)',
                height=350
            )
            
            st.plotly_chart(returns_fig, use_container_width=True)
        
        with col2:
            # Risk-Return scatter
            scatter_fig = go.Figure()
            
            for fund in funds_data:
                scatter_fig.add_trace(go.Scatter(
                    x=[fund.get('risk_score', 50)],
                    y=[fund['returns_3yr'] if pd.notna(fund['returns_3yr']) else fund['returns_1yr']],
                    mode='markers+text',
                    name=fund['scheme_name'][:20],
                    text=[fund['scheme_name'][:20]],
                    textposition='top center',
                    marker=dict(size=15)
                ))
            
            scatter_fig.update_layout(
                title='Risk vs Return',
                xaxis_title='Risk Score',
                yaxis_title='3Y Return (%)',
                height=350,
                showlegend=False
            )
            
            st.plotly_chart(scatter_fig, use_container_width=True)
        
        # Peer comparison
        st.subheader("📊 Peer Group Analysis")
        
        for fund in funds_data:
            peer_analysis = PerformanceComparator.peer_comparison(fund, analyzer.df)
            
            if peer_analysis:
                with st.expander(f"📍 {fund['scheme_name']}", expanded=False):
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Category Rank", f"{peer_analysis['rank_in_category']}/{peer_analysis['total_peers']}")
                    with col2:
                        st.metric("Return Percentile", f"{peer_analysis['returns_percentile']:.0f}th")
                    with col3:
                        st.metric("Sharpe Percentile", f"{peer_analysis['sharpe_percentile']:.0f}th")
                    with col4:
                        status = "✅ Superior" if peer_analysis['better_than_peers'] else "⚠️ Average"
                        st.metric("vs Peers", status)


def main():
    """Main application with all features"""
    
    # Header
    st.markdown('<h1 class="main-header">💰 WealthyWise Advanced</h1>', unsafe_allow_html=True)
    st.markdown("### Professional Investment Platform with Real-time Analytics")
    
    # Sidebar navigation
    st.sidebar.title("🎯 Navigation")
    
    page = st.sidebar.radio(
        "Select Module",
        [
            "🏠 Home & Live Market",
            "👤 Build Portfolio",
            "📊 Portfolio Dashboard",
            "🔬 Backtesting Lab",
            "🎯 What-If Scenarios",
            "⚖️ Fund Comparison",
            "💹 SIP Calculator",
            "📚 Learning Hub"
        ]
    )
    
    # Platform stats in sidebar
    stats = st.session_state.analyzer.get_fund_statistics()
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 Platform Stats")
    st.sidebar.metric("Funds Analyzed", stats['total_funds'])
    st.sidebar.metric("Live Updates", "Every 5 min*")
    st.sidebar.caption("*Simulated for demonstration")
    
    # Route to pages
    if page == "🏠 Home & Live Market":
        show_live_market_dashboard()
        
    elif page == "👤 Build Portfolio":
        # Inline risk profiler to avoid import issues
        st.header("📊 Build Your Investor Profile")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Personal Information")
            age = st.number_input("Your Age", min_value=MIN_AGE, max_value=MAX_AGE, value=30)
            investment_amount = st.number_input(
                "Investment Amount (₹)", 
                min_value=MIN_INVESTMENT, 
                max_value=MAX_INVESTMENT,
                value=50000,
                step=5000
            )
            horizon = st.slider(
                "Investment Horizon (Years)", 
                min_value=MIN_INVESTMENT_HORIZON,
                max_value=MAX_INVESTMENT_HORIZON,
                value=10
            )
        
        with col2:
            st.subheader("Investment Goals")
            goal = st.selectbox(
                "Primary Goal",
                options=list(INVESTMENT_GOALS.keys()),
                format_func=lambda x: INVESTMENT_GOALS[x]['name']
            )
            
            st.subheader("Risk Assessment")
            q1 = st.radio(
                "How would you react to a 20% drop?",
                ["Panic and sell", "Feel concerned", "Hold steady", "Buy more"],
                horizontal=True
            )
        
        # Calculate risk profile
        risk_scores = {
            "Panic and sell": 1, "Feel concerned": 2, "Hold steady": 3, "Buy more": 4
        }
        
        score = risk_scores[q1] + 2
        
        if score <= 3:
            risk_profile = 'conservative'
        elif score <= 5:
            risk_profile = 'moderate'
        else:
            risk_profile = 'aggressive'
        
        # Show determined risk profile
        profile_info = RISK_PROFILES[risk_profile]
        st.info(f"**Your Risk Profile: {profile_info['name']}** - {profile_info['description']}")
        
        if st.button("🎯 Generate Personalized Portfolio", type="primary", use_container_width=True):
            with st.spinner("Analyzing 800+ funds and building your optimal portfolio..."):
                portfolio = st.session_state.analyzer.recommend_funds(
                    investment_amount=investment_amount,
                    risk_profile=risk_profile,
                    age=age,
                    horizon=horizon,
                    goal=goal
                )
                
                st.session_state.user_profile = {
                    'age': age,
                    'investment_amount': investment_amount,
                    'horizon': horizon,
                    'goal': goal,
                    'risk_profile': risk_profile
                }
                st.session_state.recommended_portfolio = portfolio
                st.success("✅ Portfolio generated successfully!")
                st.rerun()
        
    elif page == "📊 Portfolio Dashboard":
        if st.session_state.recommended_portfolio is not None:
            # Show portfolio inline
            portfolio = st.session_state.recommended_portfolio
            profile = st.session_state.user_profile
            
            st.header("🎯 Your Personalized Investment Portfolio")
            
            # Portfolio summary metrics
            metrics = PortfolioAnalyzer.calculate_portfolio_metrics(portfolio)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Expected 3Y Return",
                    f"{metrics['expected_return_3yr']}% p.a.",
                    delta="Annual"
                )
            
            with col2:
                st.metric(
                    "Portfolio Sharpe Ratio",
                    f"{metrics['portfolio_sharpe']}",
                    delta="Risk-adjusted"
                )
            
            with col3:
                st.metric(
                    "Diversification Score",
                    f"{metrics['diversification_score']}/100",
                    delta="Excellent" if metrics['diversification_score'] > 70 else "Good"
                )
            
            with col4:
                st.metric(
                    "Number of Funds",
                    f"{metrics['num_funds']}",
                    delta=f"{metrics['num_categories']} Categories"
                )
            
            st.dataframe(portfolio, use_container_width=True)
        else:
            st.warning("⚠️ Please build your portfolio first from 'Build Portfolio' section")
            
    elif page == "🔬 Backtesting Lab":
        show_backtesting_lab()
        
    elif page == "🎯 What-If Scenarios":
        show_what_if_scenarios()
        
    elif page == "⚖️ Fund Comparison":
        show_performance_comparison()
        
    elif page == "💹 SIP Calculator":
        st.header("💹 SIP Calculator & Goal Planner")
        st.info("Use the What-If Scenarios section for detailed SIP analysis")
        
    elif page == "📚 Learning Hub":
        st.header("📚 Investment Education Hub")
        
        st.markdown(DISCLAIMER)
        
        with st.expander("📊 Understanding Fund Metrics", expanded=True):
            st.markdown("""
            ### Key Performance Indicators
            
            **Sharpe Ratio**: Measures risk-adjusted returns. Higher is better.
            - > 1.5: Excellent  
            - 1.0 - 1.5: Good
            - < 1.0: Average
            
            **Sortino Ratio**: Similar to Sharpe but focuses on downside risk only.
            
            **Alpha**: Excess return compared to benchmark.
            - Positive: Outperforming
            - Negative: Underperforming
            
            **Beta**: Volatility compared to market.
            - < 1: Less volatile than market
            - > 1: More volatile than market
            
            **Expense Ratio**: Annual fees charged by fund.
            - Lower is better for investors
            """)
        
        with st.expander("💡 Investment Strategies"):
            st.markdown("""
            ### Smart Investment Practices
            
            1. **Start Early**: Power of compounding works best over time
            2. **SIP over Lumpsum**: Rupee cost averaging reduces risk
            3. **Diversify**: Don't put all eggs in one basket
            4. **Long-term View**: Minimum 5 years for equity
            5. **Review Annually**: Rebalance when needed
            6. **Don't Time the Market**: Time IN the market matters
            7. **Emergency Fund First**: 6-12 months expenses
            8. **Tax Efficiency**: Use ELSS for 80C benefits
            """)


if __name__ == "__main__":
    main()
