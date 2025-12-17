"""
WealthyWise Professional - Advanced Mutual Fund Advisory Platform
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from fund_analyzer import FundAnalyzer
from portfolio_tools import SIPCalculator, GoalPlanner, PortfolioAnalyzer
from config import *
from ai_predictor import FundPerformancePredictor, integrate_ai_features
from ai_risk_assessment import EnhancedRiskProfiler

# Import the new AI Risk Persona Engine
try:
    from ai_risk_persona import get_investor_persona
    AI_PERSONA_AVAILABLE = True
except ImportError:
    AI_PERSONA_AVAILABLE = False
    st.warning("AI Risk Persona Engine not available. Using traditional risk profiling.")

# Import the WealthyWise Assistant
from wealthywise_assistant import render_chat_interface
# Import the Debug Assistant
from debug_assistant import render_debug_chat_interface

# Page configuration
st.set_page_config(
    page_title="WealthyWise Professional",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(120deg, #1e3a8a, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .fund-card {
        border: 2px solid #e5e7eb;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: transform 0.2s;
    }
    .fund-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = FundAnalyzer()
    # Initialize AI components
    predictor, optimizer = integrate_ai_features(st.session_state.analyzer, st.session_state.analyzer.df)
    st.session_state.predictor = predictor
    # Initialize AI risk profiler
    st.session_state.risk_profiler = EnhancedRiskProfiler()

if 'user_profile' not in st.session_state:
    st.session_state.user_profile = {}
if 'recommended_portfolio' not in st.session_state:
    st.session_state.recommended_portfolio = None
if 'user_persona' not in st.session_state:  # Added for AI persona
    st.session_state.user_persona = None


def show_disclaimer():
    """Display investment disclaimer"""
    with st.expander("⚠️ Important Disclaimer - Please Read", expanded=False):
        st.markdown(DISCLAIMER)


def risk_profiler():
    """Interactive risk profiling questionnaire with AI persona enhancement"""
    st.header("📊 Build Your Investor Profile")
    
    # Toggle for AI persona engine
    if AI_PERSONA_AVAILABLE:
        use_ai_persona = st.toggle("Enable AI Investor Persona Engine", value=True)
        if use_ai_persona:
            st.info("🤖 AI Persona Engine Active: Dynamic investor profiling based on behavioral patterns")
        else:
            st.info("📋 Traditional Risk Profiling: Static risk assessment based on questionnaire")
    else:
        use_ai_persona = False
        st.warning("AI Risk Persona Engine not available. Using traditional risk profiling.")
    
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
            "How would you react to a 20% drop in your portfolio?",
            ["Panic and sell", "Feel concerned", "Hold steady", "Buy more"],
            horizontal=True
        )
        
        q2 = st.radio(
            "What's more important to you?",
            ["Preserving capital", "Steady growth", "Beating inflation", "Maximum returns"],
            horizontal=True
        )
    
<<<<<<< HEAD
    # Map questionnaire answers to AI persona engine inputs
    if use_ai_persona and AI_PERSONA_AVAILABLE:
        # Map risk reaction to AI persona inputs
        risk_reaction_map = {
            "Panic and sell": "very_concerned",
            "Feel concerned": "concerned",
            "Hold steady": "neutral",
            "Buy more": "very_comfortable"
        }
        
        # Map goal importance to AI persona inputs
        goal_map = {
            "Preserving capital": "capital_preservation",
            "Steady growth": "income_generation",
            "Beating inflation": "balanced_growth",
            "Maximum returns": "aggressive_growth"
        }
        
        # Determine volatility tolerance based on risk reaction
        volatility_map = {
            "Panic and sell": "uncomfortable",
            "Feel concerned": "slightly_uncomfortable",
            "Hold steady": "neutral",
            "Buy more": "very_comfortable"
        }
        
        # Get AI persona
        try:
            persona = get_investor_persona(
                age=age,
                horizon=horizon,
                risk_reaction=risk_reaction_map[q1],
                goal=goal_map[q2],
                volatility=volatility_map[q1]
            )
            
            # Display AI persona information
            st.success(f"**🤖 AI Investor Persona: {persona['name']}**")
            st.caption(persona['explanation'])
            st.caption(f"Volatility Tolerance: {persona['volatility_tolerance']}")
            equity_range = persona['allocation_range']['equity']
            st.caption(f"Suggested Equity Allocation: {equity_range[0]}-{equity_range[1]}%")
            
            # Map persona to traditional risk profile for compatibility
            persona_risk_mapping = {
                'Capital Preservation Planner': 'conservative',
                'Income-Focused Stabilizer': 'moderate', 
                'Long-Term Growth Optimizer': 'moderate',
                'Opportunistic Risk Taker': 'aggressive'
            }
            
            ai_risk_profile = persona_risk_mapping[persona['name']]
            
            # Save persona to session state
            st.session_state.user_persona = persona
            
        except Exception as e:
            st.error(f"Error getting AI persona: {str(e)}")
            use_ai_persona = False  # Fallback to traditional method
    
    # Calculate traditional risk profile (fallback)
    if not use_ai_persona or not AI_PERSONA_AVAILABLE:
        risk_scores = {
            "Panic and sell": 1, "Feel concerned": 2, "Hold steady": 3, "Buy more": 4,
            "Preserving capital": 1, "Steady growth": 2, "Beating inflation": 3, "Maximum returns": 4
        }
        
        score = risk_scores[q1] + risk_scores[q2]
        
        if score <= 3:
            risk_profile = 'conservative'
        elif score <= 5:
            risk_profile = 'moderate'
        else:
            risk_profile = 'aggressive'
        
        # Show determined risk profile
        profile_info = RISK_PROFILES[risk_profile]
        st.info(f"**Your Risk Profile: {profile_info['name']}** - {profile_info['description']}")
=======
    # Calculate traditional risk profile
    risk_scores = {
        "Panic and sell": 1, "Feel concerned": 2, "Hold steady": 3, "Buy more": 4,
        "Preserving capital": 1, "Steady growth": 2, "Beating inflation": 3, "Maximum returns": 4
    }
    
    score = risk_scores[q1] + risk_scores[q2]
    
    if score <= 3:
        traditional_risk_profile = 'conservative'
    elif score <= 5:
        traditional_risk_profile = 'moderate'
    else:
        traditional_risk_profile = 'aggressive'
    
    # AI-powered risk assessment
    user_profile_data = {
        'age': age,
        'investment_amount': investment_amount,
        'horizon': horizon,
        'risk_reaction': q1,
        'return_preference': q2,
        'goal': goal,
        'risk_profile': traditional_risk_profile
    }
    
    # Add user profile to AI risk profiler for training
    st.session_state.risk_profiler.add_user_profile(user_profile_data)
    
    # Get AI-enhanced risk profile
    enhanced_profile = st.session_state.risk_profiler.enhanced_risk_profile(user_profile_data)
    risk_profile = enhanced_profile['final_profile']
    
    # Show determined risk profile
    profile_info = RISK_PROFILES[risk_profile]
    
    # Display both traditional and AI profiles
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Traditional Risk Profile: {RISK_PROFILES[traditional_risk_profile]['name']}**")
    with col2:
        st.success(f"**AI-Enhanced Risk Profile: {profile_info['name']}**")
    
    st.markdown(f"_Description: {profile_info['description']}_")
    
    # Show AI insights if available
    if enhanced_profile.get('ai_predicted_profile'):
        st.caption(f"_AI Confidence: {enhanced_profile['prediction_confidence']:.2f}_")
    if enhanced_profile.get('cluster_assignment') is not None:
        st.caption(f"_User Cluster: {enhanced_profile['cluster_assignment']}_")
>>>>>>> 692cca8a2124e482a55273fbecd5d5742d1c0061
    
    # Generate portfolio button
    if st.button("🎯 Generate Personalized Portfolio", type="primary", use_container_width=True):
        with st.spinner("Analyzing 800+ funds and building your optimal portfolio..."):
            # Use AI persona risk profile if available, otherwise use traditional
            final_risk_profile = ai_risk_profile if use_ai_persona and AI_PERSONA_AVAILABLE else risk_profile
            
            portfolio = st.session_state.analyzer.recommend_funds(
                investment_amount=investment_amount,
                risk_profile=final_risk_profile,
                age=age,
                horizon=horizon,
                goal=goal
            )
            
            st.session_state.user_profile = {
                'age': age,
                'investment_amount': investment_amount,
                'horizon': horizon,
                'goal': goal,
                'risk_profile': final_risk_profile,
                'ai_persona_used': use_ai_persona and AI_PERSONA_AVAILABLE
            }
            st.session_state.recommended_portfolio = portfolio
            st.success("✅ Portfolio generated successfully!")
            st.rerun()


def show_portfolio_recommendations():
    """Display recommended portfolio with detailed analysis"""
    if st.session_state.recommended_portfolio is None or len(st.session_state.recommended_portfolio) == 0:
        st.warning("Please complete the risk profiling to get recommendations.")
        return
    
    portfolio = st.session_state.recommended_portfolio
    profile = st.session_state.user_profile
    
    # Display header with persona information
    if profile.get('ai_persona_used', False) and st.session_state.user_persona:
        st.header("🎯 Your Personalized Investment Portfolio (AI Enhanced)")
        st.caption("Generated using AI Investor Persona Engine for more accurate risk profiling")
        
        # Show persona details
        persona = st.session_state.user_persona
        st.info(f"**👤 Your Investor Persona: {persona['name']}**")
        st.markdown(f"*{persona['explanation']}*")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Volatility Tolerance", persona['volatility_tolerance'])
        with col2:
            equity_range = persona['allocation_range']['equity']
            st.metric("Suggested Equity Range", f"{equity_range[0]}-{equity_range[1]}%")
        with col3:
            debt_range = persona['allocation_range']['debt']
            st.metric("Suggested Debt Range", f"{debt_range[0]}-{debt_range[1]}%")
    else:
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
    
    # Allocation pie chart
    st.subheader("📊 Asset Allocation")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        category_allocation = portfolio.groupby('category')['allocation_percentage'].sum().reset_index()
        fig = px.pie(
            category_allocation,
            values='allocation_percentage',
            names='category',
            title='Portfolio Distribution by Category',
            color_discrete_sequence=px.colors.sequential.Viridis
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Allocation Summary")
        for _, row in category_allocation.iterrows():
            st.markdown(f"**{row['category']}**: {row['allocation_percentage']:.1f}%")
        st.markdown(f"**Total Investment**: ₹{profile['investment_amount']:,.0f}")
    
    # Detailed fund list
    st.subheader("📋 Recommended Funds")
    
    for idx, fund in portfolio.iterrows():
        with st.container():
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.markdown(f"### {fund['scheme_name']}")
                st.caption(f"{fund['amc_name']} | {fund['category']}")
                st.caption(f"Fund Manager: {fund['fund_manager']}")
            
            with col2:
                st.metric("Allocation", f"{fund['allocation_percentage']:.1f}%")
                st.metric("Amount", f"₹{fund['investment_amount']:,.0f}")
            
            with col3:
                st.metric("Quality Score", f"{fund['quality_score']:.0f}/100")
                rating_stars = "⭐" * int(fund['rating']) if pd.notna(fund['rating']) else "N/A"
                st.markdown(f"**Rating**: {rating_stars}")
            
            # Expandable details
            with st.expander("View Detailed Metrics"):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("1Y Return", f"{fund['returns_1yr']:.1f}%")
                    st.metric("Sharpe Ratio", f"{fund['sharpe']:.2f}")
                
                with col2:
                    st.metric("3Y Return", f"{fund['returns_3yr']:.1f}%")
                    st.metric("Alpha", f"{fund['alpha']:.2f}")
                
                with col3:
                    st.metric("5Y Return", f"{fund['returns_5yr']:.1f}%" if pd.notna(fund['returns_5yr']) else "N/A")
                    st.metric("Expense Ratio", f"{fund['expense_ratio']:.2f}%")
                
                with col4:
                    st.metric("Risk Score", f"{fund['risk_score']:.0f}/100")
            
            st.divider()


def sip_calculator_page():
    """SIP calculator with projections"""
    st.header("💹 SIP Calculator & Goal Planner")
    
    tab1, tab2, tab3 = st.tabs(["SIP Calculator", "Goal Planning", "Tax Calculator"])
    
    with tab1:
        st.subheader("Calculate Your SIP Returns")
        
        col1, col2 = st.columns(2)
        
        with col1:
            monthly_sip = st.number_input("Monthly SIP Amount (₹)", min_value=500, value=10000, step=500)
            years = st.slider("Investment Period (Years)", 1, 30, 10)
            expected_return = st.slider("Expected Annual Return (%)", 5.0, 20.0, 12.0, 0.5)
            step_up = st.selectbox("Annual Step-up (%)", [0] + SIP_STEP_UP_OPTIONS)
        
        if st.button("Calculate SIP Returns"):
            result = SIPCalculator.calculate_future_value(
                monthly_sip, expected_return / 100, years, step_up
            )
            
            with col2:
                st.metric("Future Value", f"₹{result['future_value']:,.0f}")
                st.metric("Total Invested", f"₹{result['total_invested']:,.0f}")
                st.metric("Total Gains", f"₹{result['total_gains']:,.0f}")
                st.metric("Returns", f"{result['returns_percentage']:.1f}%")
            
            # Yearly breakdown chart
            df_breakdown = pd.DataFrame(result['yearly_breakdown'])
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df_breakdown['year'],
                y=df_breakdown['total_invested'],
                name='Invested',
                marker_color='lightblue'
            ))
            fig.add_trace(go.Bar(
                x=df_breakdown['year'],
                y=df_breakdown['gains'],
                name='Gains',
                marker_color='lightgreen'
            ))
            
            fig.update_layout(
                title='Year-wise Growth',
                barmode='stack',
                xaxis_title='Year',
                yaxis_title='Amount (₹)'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Goal-Based Planning")
        
        goal_type = st.selectbox(
            "Select Your Goal",
            ['retirement', 'child_education', 'home_purchase'],
            format_func=lambda x: INVESTMENT_GOALS[x]['name']
        )
        
        if goal_type == 'retirement':
            col1, col2 = st.columns(2)
            with col1:
                current_age = st.number_input("Current Age", 25, 60, 30)
                retirement_age = st.number_input("Retirement Age", current_age + 5, 70, 60)
            with col2:
                monthly_expenses = st.number_input("Current Monthly Expenses (₹)", 10000, 500000, 50000)
            
            if st.button("Calculate Retirement Plan"):
                plan = GoalPlanner.plan_retirement(current_age, retirement_age, monthly_expenses)
                
                st.success(f"**Corpus Needed**: ₹{plan['corpus_needed']:,.0f}")
                st.info(f"**Required Monthly SIP**: ₹{plan['required_monthly_sip']:,.0f}")
                st.warning(f"Future monthly expenses at retirement: ₹{plan['future_monthly_expenses']:,.0f}")
        
        elif goal_type == 'child_education':
            col1, col2 = st.columns(2)
            with col1:
                years_until = st.number_input("Years Until Needed", 1, 20, 10)
            with col2:
                current_cost = st.number_input("Current Education Cost (₹)", 100000, 10000000, 2000000)
            
            if st.button("Calculate Education Plan"):
                plan = GoalPlanner.plan_child_education(years_until, current_cost)
                
                st.success(f"**Future Cost**: ₹{plan['future_cost']:,.0f}")
                st.info(f"**Required Monthly SIP**: ₹{plan['required_monthly_sip']:,.0f}")
                st.warning(f"**OR Lumpsum Today**: ₹{plan['required_lumpsum_today']:,.0f}")
    
    with tab3:
        st.subheader("Tax Impact Calculator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            gains = st.number_input("Capital Gains (₹)", min_value=0, value=100000)
            inv_type = st.selectbox("Investment Type", ["Equity", "Debt"])
            holding_years = st.number_input("Holding Period (Years)", 0.1, 10.0, 1.5)
        
        if st.button("Calculate Tax"):
            tax_info = SIPCalculator.calculate_tax_impact(gains, inv_type, holding_years)
            
            with col2:
                st.metric("Gross Gains", f"₹{tax_info['gross_gains']:,.0f}")
                st.metric("Tax Amount", f"₹{tax_info['tax_amount']:,.0f}")
                st.metric("Net Gains", f"₹{tax_info['net_gains']:,.0f}")
                st.info(f"Tax Type: {tax_info['tax_type']}")
                st.warning(f"Effective Tax Rate: {tax_info['effective_tax_rate']:.1f}%")


def fund_explorer():
    """Explore and compare funds"""
    st.header("🔍 Fund Explorer & Comparison")
    
    analyzer = st.session_state.analyzer
    
    tab1, tab2 = st.tabs(["Top Performers", "Fund Comparison"])
    
    with tab1:
        col1, col2 = st.columns([1, 3])
        
        with col1:
            category = st.selectbox(
                "Select Category",
                ['All'] + sorted(analyzer.df['category'].unique().tolist())
            )
            n_funds = st.slider("Number of Funds", 5, 50, 10)
        
        cat_filter = None if category == 'All' else category
        top_funds = analyzer.get_top_performers(cat_filter, n_funds)
        
        with col2:
            st.dataframe(
                top_funds.style.background_gradient(subset=['quality_score'], cmap='RdYlGn'),
                use_container_width=True,
                height=600
            )
    
    with tab2:
        st.subheader("Compare Funds Side-by-Side")
        
        all_schemes = analyzer.df['scheme_name'].tolist()
        selected_funds = st.multiselect(
            "Select Funds to Compare (max 5)",
            all_schemes,
            max_selections=5
        )
        
        if len(selected_funds) > 1:
            comparison = analyzer.compare_funds(selected_funds)
            
            # Display comparison
            st.dataframe(comparison, use_container_width=True)
            
            # Visual comparison
            metrics_to_plot = ['quality_score', 'sharpe', 'alpha', 'returns_3yr']
            
            fig = go.Figure()
            
            for metric in metrics_to_plot:
                fig.add_trace(go.Bar(
                    name=metric.replace('_', ' ').title(),
                    x=comparison['scheme_name'],
                    y=comparison[metric],
                ))
            
            fig.update_layout(barmode='group', title='Performance Comparison')
            st.plotly_chart(fig, use_container_width=True)


def main():
    st.markdown('<h1 class="main-header">WealthyWise Professional</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://placehold.co/250x100?text=WealthyWise+Logo", use_column_width=True)
        st.markdown("---")
        
        # Define pages with emojis
        pages = [
            "🏠 Home Dashboard",
            "📊 Risk Profiler",
            "🎯 Portfolio Recommendations",
            "💹 SIP & Goal Calculator",
            "🔍 Fund Explorer",
            "🎓 WealthyWise Assistant",
            "🐛 Debug Assistant",  # Added debug assistant for troubleshooting
            "📚 Learning Center"
        ]
        
        page = st.selectbox(
            "Navigate",
            pages,
            format_func=lambda x: x.split(" ", 1)[1]  # Remove emoji for cleaner display
        )
        
        st.markdown("---")
        show_disclaimer()
    
    # Page routing - use the full page name including emoji
    if page == "🏠 Home Dashboard":
        st.header("Welcome to WealthyWise Professional")
        st.markdown("""
        Your AI-powered investment advisory platform for smarter mutual fund decisions.
        
        ### Key Features:
        - 🤖 **AI Investor Persona Engine**: Dynamic risk profiling based on behavioral patterns
        - 📊 **Smart Fund Recommendations**: Personalized fund suggestions based on your profile
        - 💹 **SIP Calculators**: Plan your investments with precision
        - 🔍 **Fund Explorer**: Compare and analyze 800+ mutual funds
        - 🎓 **WealthyWise Assistant**: Educational chatbot for investment guidance
        """)
        
        # Show user profile if exists
        if st.session_state.user_profile:
            st.subheader("Your Current Profile")
            profile = st.session_state.user_profile
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Age", f"{profile['age']} years")
            with col2:
                st.metric("Investment Horizon", f"{profile['horizon']} years")
            with col3:
                st.metric("Investment Amount", f"₹{profile['investment_amount']:,.0f}")
    
    elif page == "📊 Risk Profiler":
        risk_profiler()
    
    elif page == "🎯 Portfolio Recommendations":
        if st.session_state.recommended_portfolio is not None:
            show_portfolio_recommendations()
        else:
            st.info("Please complete the risk profiling first to get portfolio recommendations.")
            if st.button("Go to Risk Profiler"):
                st.session_state.current_page = "Risk Profiler"
                st.rerun()
    
    elif page == "💹 SIP & Goal Calculator":
        sip_calculator_page()
    
    elif page == "🔍 Fund Explorer":
        fund_explorer()
    
    elif page == "🎓 WealthyWise Assistant":
        render_chat_interface()
    
    elif page == "🐛 Debug Assistant":
        render_debug_chat_interface()
    
    elif page == "📚 Learning Center":
        st.header("📚 Investment Education Hub")
        
        with st.expander("📊 Understanding Fund Metrics"):
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
        
        with st.expander("🎯 Goal-Based Investing"):
            st.markdown("""
            ### Match Goals with Investment Horizon
            
            **Short-term (< 3 years)**
            - Emergency fund
            - Vacation planning
            - Use: Debt funds, liquid funds
            
            **Medium-term (3-7 years)**
            - Home down payment
            - Car purchase
            - Use: Balanced/hybrid funds
            
            **Long-term (> 7 years)**
            - Retirement
            - Child education
            - Wealth creation
            - Use: Equity funds, aggressive hybrid
            """)

if __name__ == "__main__":
    main()
