"""
Demonstration of AI Risk Persona Engine Integration

This script shows how the AI Risk Persona Engine can be integrated into the 
WealthyWise Professional platform.
"""

import streamlit as st
from fund_analyzer import FundAnalyzer
from portfolio_tools import PortfolioAnalyzer

# Try to import the AI persona engine
try:
    from ai_risk_persona import get_investor_persona
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    st.error("AI Risk Persona Engine not available. Please check installation.")

def main():
    st.set_page_config(page_title="AI Persona Engine Demo", page_icon="🤖")
    st.title("🤖 AI Investor Persona Engine Demo")
    
    if not AI_AVAILABLE:
        st.warning("AI components not available. Showing traditional approach only.")
        return
    
    st.markdown("""
    This demo shows how the AI Risk Persona Engine enhances the WealthyWise Professional platform
    by providing dynamic investor profiling based on behavioral patterns rather than static questionnaires.
    """)
    
    # Sidebar for inputs
    st.sidebar.header("Investor Profile")
    age = st.sidebar.number_input("Age", 18, 80, 35)
    horizon = st.sidebar.number_input("Investment Horizon (years)", 1, 40, 15)
    
    risk_reaction = st.sidebar.selectbox(
        "Reaction to Market Downturns",
        ["very_concerned", "concerned", "neutral", "comfortable", "very_comfortable"]
    )
    
    goal = st.sidebar.selectbox(
        "Primary Financial Goal",
        ["capital_preservation", "income_generation", "balanced_growth", "aggressive_growth"]
    )
    
    volatility = st.sidebar.selectbox(
        "Comfort with Portfolio Fluctuations",
        ["uncomfortable", "slightly_uncomfortable", "neutral", "comfortable", "very_comfortable"]
    )
    
    # Get AI persona
    if st.sidebar.button("🔍 Analyze Investor Persona"):
        try:
            persona = get_investor_persona(age, horizon, risk_reaction, goal, volatility)
            
            # Display persona information
            st.subheader(f"👤 {persona['name']}")
            st.write(persona['explanation'])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Volatility Tolerance", persona['volatility_tolerance'])
            with col2:
                equity_range = persona['allocation_range']['equity']
                st.metric("Suggested Equity Allocation", f"{equity_range[0]}-{equity_range[1]}%")
            with col3:
                debt_range = persona['allocation_range']['debt']
                st.metric("Suggested Debt Allocation", f"{debt_range[0]}-{debt_range[1]}%")
            
            # Show how this maps to traditional risk profiles
            st.subheader("🔄 Compatibility Mapping")
            persona_risk_mapping = {
                'Capital Preservation Planner': 'Conservative',
                'Income-Focused Stabilizer': 'Moderate', 
                'Long-Term Growth Optimizer': 'Moderate',
                'Opportunistic Risk Taker': 'Aggressive'
            }
            
            traditional_profile = persona_risk_mapping[persona['name']]
            st.info(f"Maps to traditional risk profile: **{traditional_profile}**")
            
            # Demonstrate enhanced fund recommendation
            st.subheader("💰 Enhanced Fund Recommendations")
            st.write("Using the AI persona to guide fund selection:")
            
            # Initialize analyzer
            analyzer = FundAnalyzer()
            
            # Map persona to traditional risk profile for compatibility
            risk_profile_mapping = {
                'Capital Preservation Planner': 'conservative',
                'Income-Focused Stabilizer': 'moderate', 
                'Long-Term Growth Optimizer': 'moderate',
                'Opportunistic Risk Taker': 'aggressive'
            }
            
            mapped_risk_profile = risk_profile_mapping[persona['name']]
            
            # Get recommendations
            recommendations = analyzer.recommend_funds(
                investment_amount=100000,
                risk_profile=mapped_risk_profile,
                age=age,
                horizon=horizon
            )
            
            if len(recommendations) > 0:
                st.write(f"Recommended {len(recommendations)} funds based on your persona:")
                
                # Show portfolio metrics
                metrics = PortfolioAnalyzer.calculate_portfolio_metrics(recommendations)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Expected Return", f"{metrics['expected_return_3yr']}%")
                with col2:
                    st.metric("Sharpe Ratio", f"{metrics['portfolio_sharpe']}")
                with col3:
                    st.metric("Diversification", f"{metrics['diversification_score']}/100")
                with col4:
                    st.metric("Number of Funds", f"{metrics['num_funds']}")
                
                # Show fund allocation
                st.subheader("📊 Portfolio Allocation")
                category_allocation = recommendations.groupby('category')['allocation_percentage'].sum()
                for category, allocation in category_allocation.items():
                    st.progress(allocation/100)
                    st.write(f"{category}: {allocation:.1f}%")
            else:
                st.warning("No fund recommendations available for this profile.")
                
        except Exception as e:
            st.error(f"Error analyzing persona: {str(e)}")
    
    # Show technical details
    with st.expander("🔬 Technical Details"):
        st.markdown("""
        ### How the AI Persona Engine Works
        
        1. **Data Preprocessing**: Investor profile data is encoded into numerical features
        2. **Feature Scaling**: StandardScaler normalizes the feature values
        3. **Clustering**: KMeans algorithm groups investors into 4 behavioral personas
        4. **Mapping**: Clusters are mapped to meaningful persona names and characteristics
        5. **Integration**: Persona information guides fund recommendations
        
        ### Benefits Over Traditional Approaches
        
        - **Dynamic Profiling**: Adapts to nuanced behavioral patterns
        - **Data-Driven**: Based on actual investor clustering rather than fixed rules
        - **Extensible**: Easy to add new personas or features
        - **Compatible**: Seamlessly integrates with existing risk profiles
        """)

if __name__ == "__main__":
    main()