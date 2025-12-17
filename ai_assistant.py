"""
AI Financial Assistant for WealthyWise
Provides intelligent responses to user financial queries
"""
import streamlit as st
import pandas as pd
import re
from typing import Dict, List, Tuple


class FinancialAIAssistant:
    """AI assistant for financial guidance"""
    
    def __init__(self):
        # Predefined responses for common queries
        self.knowledge_base = {
            "sip": {
                "keywords": ["sip", "systematic investment plan", "monthly investment"],
                "response": """
                💡 **Systematic Investment Plan (SIP)** is a disciplined way to invest in mutual funds.
                
                **Benefits of SIP:**
                1. **Rupee Cost Averaging** - Reduces impact of market volatility
                2. **Discipline** - Regular investing habit
                3. **Power of Compounding** - Long-term wealth creation
                4. **Flexibility** - Start with as low as ₹500/month
                
                **Recommended Approach:**
                - Start early for maximum benefits
                - Increase SIP amount annually (10-15%)
                - Stay invested for at least 5-7 years
                """
            },
            "risk": {
                "keywords": ["risk", "volatility", "fluctuation"],
                "response": """
                ⚠️ **Understanding Investment Risk**
                
                **Types of Risks:**
                1. **Market Risk** - Prices fluctuate with markets
                2. **Credit Risk** - Issuer may default (more in corporate bonds)
                3. **Interest Rate Risk** - Bond prices fall when rates rise
                4. **Inflation Risk** - Returns may not beat inflation
                
                **Risk Management:**
                - Diversification across asset classes
                - Asset allocation based on age and goals
                - Regular portfolio review and rebalancing
                - Long-term perspective to smooth volatility
                """
            },
            "diversification": {
                "keywords": ["diversification", "diversify", "spread risk"],
                "response": """
                🌈 **Importance of Diversification**
                
                **Why Diversify?**
                - Reduces portfolio risk
                - Minimizes impact of poor performers
                - Captures growth across sectors/markets
                
                **How to Diversify:**
                1. **Asset Class** - Equity, Debt, Gold, REITs
                2. **Sector** - IT, Banking, Pharma, FMCG, etc.
                3. **Geography** - Domestic and International funds
                4. **Market Cap** - Large-cap, Mid-cap, Small-cap
                
                **Golden Rule:** Don't put all eggs in one basket!
                """
            },
            "emergency fund": {
                "keywords": ["emergency", "savings", "liquid"],
                "response": """
                🛡️ **Emergency Fund Importance**
                
                **What is an Emergency Fund?**
                - 6-12 months of living expenses
                - Easily accessible (liquid funds, savings account)
                - Separate from investment portfolio
                
                **Why You Need It:**
                - Job loss or income disruption
                - Medical emergencies
                - Unexpected expenses
                - Avoid selling investments during downturns
                
                **Where to Keep:**
                - Liquid funds (better returns than savings)
                - Ultra-short duration funds
                - High-yield savings accounts
                """
            },
            "tax": {
                "keywords": ["tax", "elss", "80c", "deduction"],
                "response": """
                💰 **Tax Saving Investments (Section 80C)**
                
                **ELSS (Equity Linked Savings Scheme):**
                - Lock-in period: 3 years (shortest among 80C options)
                - Potential returns: 12-15% annually
                - Tax deduction up to ₹1.5 lakh/year
                
                **Other 80C Options:**
                - PPF (15-year lock-in, ~7-8% returns)
                - NSC (5-year lock-in, ~7-8% returns)
                - FDs (5-year lock-in, ~6-7% returns)
                - ULIPs (5-year lock-in, variable returns)
                
                **Smart Strategy:**
                - Maximize ELSS allocation for better returns
                - Balance with other instruments based on risk profile
                """
            },
            "retirement": {
                "keywords": ["retirement", "pension", "nest egg"],
                "response": """
                🏖️ **Retirement Planning Essentials**
                
                **Start Early Principle:**
                - Starting at 25 vs 35 can make 5-7x difference
                - Power of compounding works best over long periods
                
                **Asset Allocation Strategy:**
                - Age 25-35: 80-90% Equity, 10-20% Debt
                - Age 35-50: 60-80% Equity, 20-40% Debt
                - Age 50+: Gradually shift to conservative allocation
                
                **Key Steps:**
                1. Calculate retirement corpus needed
                2. Increase contribution annually
                3. Review and rebalance regularly
                4. Consider annuities for guaranteed income
                """
            }
        }
    
    def preprocess_query(self, query: str) -> str:
        """Clean and normalize user query"""
        return query.lower().strip()
    
    def find_best_match(self, query: str) -> str:
        """Find the best matching response category"""
        processed_query = self.preprocess_query(query)
        
        best_match = None
        highest_score = 0
        
        for category, data in self.knowledge_base.items():
            score = 0
            for keyword in data["keywords"]:
                if keyword in processed_query:
                    score += 1
            
            # Bonus for exact phrase matches
            if category.replace(" ", "") in processed_query.replace(" ", ""):
                score += 2
            
            if score > highest_score:
                highest_score = score
                best_match = category
        
        return best_match if highest_score > 0 else None
    
    def get_response(self, query: str) -> str:
        """Generate response for user query"""
        category = self.find_best_match(query)
        
        if category:
            return self.knowledge_base[category]["response"]
        else:
            return """
            🤖 **I'm still learning!** I couldn't find a specific answer to your question.
            
            For personalized advice, I recommend:
            1. Completing your risk profile assessment
            2. Consulting with a certified financial advisor
            3. Reviewing our educational resources
            
            Alternatively, you can ask about:
            - SIP benefits and strategies
            - Risk management techniques
            - Portfolio diversification
            - Tax-saving investments
            - Retirement planning
            """
    
    def render_chat_interface(self):
        """Render the chat interface in Streamlit"""
        st.subheader("🤖 AI Financial Assistant")
        st.markdown("*Ask me anything about investments, SIPs, taxes, or retirement planning!*")
        
        # Initialize session state for chat
        if "ai_messages" not in st.session_state:
            st.session_state.ai_messages = [
                {"role": "assistant", "content": "Hello! I'm your AI financial assistant. How can I help you with your investments today?"}
            ]
        
        # Display chat messages
        for message in st.session_state.ai_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask about investments, SIPs, taxes, etc."):
            # Add user message
            st.session_state.ai_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate and add assistant response
            response = self.get_response(prompt)
            st.session_state.ai_messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)


def integrate_ai_assistant():
    """Function to integrate AI assistant into the main app"""
    assistant = FinancialAIAssistant()
    assistant.render_chat_interface()


if __name__ == "__main__":
    # Example usage
    assistant = FinancialAIAssistant()
    
    # Test some queries
    test_queries = [
        "What is SIP?",
        "How to manage investment risk?",
        "Why should I diversify my portfolio?",
        "How much emergency fund should I keep?",
        "Tell me about ELSS funds for tax saving"
    ]
    
    for query in test_queries:
        print(f"\nQ: {query}")
        print(f"A: {assistant.get_response(query)}")
        print("-" * 50)